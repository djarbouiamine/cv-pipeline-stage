"""Remove a CV from Elasticsearch, cache, uploads, and output files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set

from cv_cache import CVCache
from cv_saver import save_to_json, save_to_excel

CVS_DIR = "cvs"
UPLOAD_DIR = "cvs_uploads"
JSON_PATH = "output/cvs_data.json"
INDEX_NAME = "cvs"


def _basename(path_or_name: str) -> str:
    return os.path.basename((path_or_name or "").replace("\\", "/"))


def _normalize_source(source: Dict[str, Any]) -> Dict[str, str]:
    return {
        "email": (source.get("email") or "").strip().lower(),
        "nom": (source.get("nom") or "").strip(),
        "phone": (source.get("telephone") or source.get("phone") or "").strip(),
        "filename": _basename(source.get("filename") or ""),
    }


def _remove_from_cache(cache: CVCache, doc_id: str, source: Dict[str, Any]) -> tuple[bool, List[Dict[str, Any]]]:
    meta = _normalize_source(source)
    removed_entries = cache.delete_matching(
        doc_hash=str(doc_id or ""),
        email=meta["email"],
        nom=meta["nom"],
        phone=meta["phone"],
        filename=meta["filename"],
    )
    return bool(removed_entries), removed_entries


def _remove_pdf_files(source: Dict[str, Any], cache_entries: List[Dict[str, Any]]) -> bool:
    """Delete PDF files from cvs/, cvs_uploads/, and any recorded source_path."""
    candidates: Set[str] = set()
    meta = _normalize_source(source)

    if meta["filename"]:
        candidates.add(meta["filename"])
    raw_filename = (source.get("filename") or "").replace("\\", "/")
    if raw_filename:
        candidates.add(_basename(raw_filename))

    for entry in cache_entries:
        source_path = (entry.get("source_path") or "").replace("\\", "/")
        if source_path:
            candidates.add(source_path)
            candidates.add(_basename(source_path))

    removed = False
    checked_paths: Set[str] = set()

    for candidate in candidates:
        path_options = [candidate]
        if not os.path.isabs(candidate):
            path_options.extend([
                os.path.join(CVS_DIR, _basename(candidate)),
                os.path.join(UPLOAD_DIR, _basename(candidate)),
            ])

        for path in path_options:
            norm = os.path.normpath(path)
            if norm in checked_paths:
                continue
            checked_paths.add(norm)
            if os.path.isfile(norm):
                os.remove(norm)
                removed = True

    return removed


def _remove_from_output_files(source: Dict[str, Any]) -> bool:
    if not os.path.exists(JSON_PATH):
        return False

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    meta = _normalize_source(source)
    filename = meta["filename"]
    email = meta["email"]
    nom = meta["nom"]

    def _should_keep(entry: Dict[str, Any]) -> bool:
        entry_data = entry.get("data") or {}
        entry_filename = _basename(entry.get("filename") or "")
        entry_email = (entry_data.get("email") or "").strip().lower()
        entry_nom = (entry_data.get("nom") or "").strip()
        if filename and entry_filename == filename:
            return False
        if email and entry_email == email:
            return False
        if nom and entry_nom == nom:
            return False
        return True

    new_results = [entry for entry in all_results if _should_keep(entry)]
    if len(new_results) == len(all_results):
        return False

    save_to_json(new_results, JSON_PATH)
    save_to_excel(new_results)
    return True


def _collect_es_doc_ids(es, doc_id: str, source: Dict[str, Any]) -> Set[str]:
    """Find Elasticsearch document IDs matching hash, email, name, or filename."""
    ids: Set[str] = set()
    if doc_id:
        ids.add(str(doc_id))

    if not hasattr(es, "search"):
        return ids

    meta = _normalize_source(source)
    email = meta["email"]
    nom = meta["nom"]
    filename = meta["filename"]

    should: List[Dict[str, Any]] = []
    if email:
        should.append({"term": {"email": email}})
    if nom:
        should.append({"match_phrase": {"nom": nom}})
    if filename:
        should.append({"wildcard": {"filename": f"*{filename}"}})

    if not should:
        return ids

    try:
        resp = es.search(
            index=INDEX_NAME,
            size=100,
            query={"bool": {"should": should, "minimum_should_match": 1}},
            _source=False,
        )
        for hit in resp.get("hits", {}).get("hits", []):
            hit_id = hit.get("_id")
            if hit_id is not None:
                ids.add(str(hit_id))
    except Exception:
        pass

    return ids


def _delete_from_elasticsearch(es, doc_ids: Set[str]) -> int:
    """Delete documents from ES. Returns count of successful deletions."""
    if not hasattr(es, "delete") or not doc_ids:
        return 0

    deleted = 0
    for eid in doc_ids:
        try:
            es.delete(index=INDEX_NAME, id=eid)
            deleted += 1
        except Exception:
            try:
                if hasattr(es, "exists") and es.exists(index=INDEX_NAME, id=eid):
                    es.delete(index=INDEX_NAME, id=eid, refresh=True)
                    deleted += 1
            except Exception:
                continue

    if deleted and hasattr(es, "indices"):
        try:
            es.indices.refresh(index=INDEX_NAME)
        except Exception:
            pass

    return deleted


def remove_cv(es, cache: CVCache, doc_id: str, source: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a CV everywhere it may be stored."""
    cache_removed, cache_entries = _remove_from_cache(cache, doc_id, source)

    es_ids = _collect_es_doc_ids(es, doc_id, source)
    for entry in cache_entries:
        entry_hash = entry.get("hash")
        if entry_hash:
            es_ids.add(str(entry_hash))
    es_deleted = _delete_from_elasticsearch(es, es_ids)

    pdf_removed = _remove_pdf_files(source, cache_entries)

    report = {
        "elasticsearch": es_deleted > 0,
        "elasticsearch_count": es_deleted,
        "cache": cache_removed,
        "cache_count": len(cache_entries),
        "pdf": pdf_removed,
        "output": _remove_from_output_files(source),
    }
    report["success"] = any([
        report["elasticsearch"],
        report["cache"],
        report["pdf"],
        report["output"],
    ])
    return report
