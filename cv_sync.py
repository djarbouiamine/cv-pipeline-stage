"""
cv_sync.py — Synchronisation du cache et des donnees JSON avec les PDFs presents sur disque.
==============================================================================================

Ce module detecte les entrees "orphelines" : CVs encore presents dans
  - output/cv_cache.json
  - output/cvs_data.json
  - l index Elasticsearch "cvs"
mais dont le fichier PDF source n existe plus dans cvs/ ou cvs_uploads/.

Usage programmatique :
    from cv_sync import sync_orphans
    report = sync_orphans(es=es_client)   # es facultatif

Usage CLI :
    python cv_sync.py
    python cv_sync.py --dry-run
"""

from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any, Set

ROOT = os.path.dirname(os.path.abspath(__file__))
CV_DIRS = [
    os.path.join(ROOT, "cvs"),
    os.path.join(ROOT, "cvs_uploads"),
]
CACHE_PATH = os.path.join(ROOT, "output", "cv_cache.json")
DATA_PATH  = os.path.join(ROOT, "output", "cvs_data.json")
INDEX_NAME = "cvs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_pdf_filenames() -> Set[str]:
    """Return basenames of every PDF that still exists in cvs/ and cvs_uploads/."""
    found: Set[str] = set()
    for folder in CV_DIRS:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".pdf"):
                found.add(fname)
    return found


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_json(path: str, data: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _entry_filename(entry: dict) -> str:
    """Extract the bare filename from a cache or data entry."""
    for key in ("source_path", "filename"):
        val = entry.get(key, "")
        if val:
            return os.path.basename(val.replace("\\", "/"))
    return ""


# ---------------------------------------------------------------------------
# Core sync function
# ---------------------------------------------------------------------------

def sync_orphans(
    es=None,
    cache_path: str = CACHE_PATH,
    data_path: str  = DATA_PATH,
    index_name: str = INDEX_NAME,
    dry_run: bool   = False,
) -> Dict[str, Any]:
    """
    Detect and remove orphaned entries whose PDF no longer exists on disk.

    Parameters
    ----------
    es        : Elasticsearch client (optional). If None, ES pruning is skipped.
    cache_path: Path to output/cv_cache.json
    data_path : Path to output/cvs_data.json
    index_name: Elasticsearch index name
    dry_run   : If True, report what would be removed but do not write anything.

    Returns
    -------
    dict with keys:
        existing_pdfs   : set of PDF basenames still on disk
        orphaned_cache  : list of orphaned filenames from cache
        orphaned_data   : list of orphaned filenames from cvs_data
        orphaned_es     : list of ES document IDs removed
        removed_cache   : int  (0 if dry_run)
        removed_data    : int  (0 if dry_run)
        removed_es      : int  (0 if dry_run)
        es_skipped      : bool (True when es is None or not available)
    """
    existing_pdfs = _all_pdf_filenames()

    # 1. Prune cv_cache.json
    cache_entries: List[dict] = _load_json(cache_path)
    keep_cache: List[dict] = []
    orphaned_cache: List[str] = []
    for entry in cache_entries:
        fname = _entry_filename(entry)
        if fname and fname not in existing_pdfs:
            orphaned_cache.append(fname)
        else:
            keep_cache.append(entry)

    if orphaned_cache and not dry_run:
        _save_json(cache_path, keep_cache)

    # 2. Prune cvs_data.json
    data_entries: List[dict] = _load_json(data_path)
    keep_data: List[dict] = []
    orphaned_data: List[str] = []
    for entry in data_entries:
        fname = _entry_filename(entry)
        if fname and fname not in existing_pdfs:
            orphaned_data.append(fname)
        else:
            keep_data.append(entry)

    if orphaned_data and not dry_run:
        _save_json(data_path, keep_data)

    # 3. Prune Elasticsearch
    orphaned_es: List[str] = []
    es_skipped = es is None

    if es is not None:
        try:
            resp = es.search(
                index=index_name,
                body={"query": {"match_all": {}}, "_source": ["filename"], "size": 5000},
            )
            hits = resp.get("hits", {}).get("hits", [])
            for hit in hits:
                es_fname = os.path.basename(
                    (hit.get("_source", {}).get("filename") or "").replace("\\", "/")
                )
                if es_fname and es_fname not in existing_pdfs:
                    orphaned_es.append(hit["_id"])
                    if not dry_run:
                        try:
                            es.delete(index=index_name, id=hit["_id"])
                        except Exception:
                            pass
        except Exception as e:
            print(f"  [WARN] Elasticsearch non disponible pour la synchronisation : {e}")
            es_skipped = True

    return {
        "existing_pdfs":  existing_pdfs,
        "orphaned_cache": orphaned_cache,
        "orphaned_data":  orphaned_data,
        "orphaned_es":    orphaned_es,
        "removed_cache":  0 if dry_run else len(orphaned_cache),
        "removed_data":   0 if dry_run else len(orphaned_data),
        "removed_es":     0 if dry_run else len(orphaned_es),
        "es_skipped":     es_skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Synchronise le cache et les JSON avec les PDFs presents sur disque."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les orphelins sans rien supprimer."
    )
    args = parser.parse_args()

    es_client = None
    try:
        from elasticsearch import Elasticsearch as _ES
        _es = _ES("http://localhost:9200")
        _es.cluster.health(timeout="3s")
        es_client = _es
        print("  Elasticsearch connecte.")
    except Exception:
        print("  Elasticsearch non disponible -- synchronisation ES ignoree.")

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Synchronisation en cours...\n")
    report = sync_orphans(es=es_client, dry_run=args.dry_run)

    print(f"  PDFs presents sur disque    : {len(report['existing_pdfs'])}")
    print(f"  Orphelins cache             : {len(report['orphaned_cache'])}")
    print(f"  Orphelins cvs_data          : {len(report['orphaned_data'])}")
    print(f"  Orphelins Elasticsearch     : {len(report['orphaned_es'])}")

    if report["orphaned_cache"]:
        print("\n  Fichiers purges du cache :")
        for f in report["orphaned_cache"]:
            print(f"    - {f}")

    if not args.dry_run:
        print(
            f"\n  Supprimes : {report['removed_cache']} cache | "
            f"{report['removed_data']} data | "
            f"{report['removed_es']} ES"
        )
    else:
        print("\n  (Dry-run : aucune modification effectuee)")
