"""
cv_deduplication.py
Détection de doublons multi-niveaux pour les CVs :
  - Niveau 1 : hash exact du texte brut
  - Niveau 2 : même candidat (email / téléphone) + distinction mise à jour vs doublon
  - Niveau 3 : similarité sémantique (embeddings)

Expose aussi check_single_cv_duplicates() pour la vérification interactive
d'un seul CV (page "Ajouter un CV").
"""

import os
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from cv_extractor import embedding_model

# ---------------------------------------------------------------------------
# Constantes (surchargeables via .env)
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

DEFAULT_SIMILARITY_THRESHOLD = _env_float("DEDUP_SIMILARITY_THRESHOLD", 0.90)
UPDATE_SIMILARITY_THRESHOLD  = _env_float("DEDUP_UPDATE_THRESHOLD", 0.98)


def _normalize_phone(phone: str) -> str:
    return "".join(ch for ch in (phone or "") if ch.isdigit())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    """SHA-256 du texte brut (encodé en UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs numpy."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Détection de doublons en batch (traitement d'un lot de CVs)
# ---------------------------------------------------------------------------

def detect_duplicates(
    cvs: List[Dict[str, Any]],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    update_similarity_threshold: float = 0.98,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    update_similarity_threshold : au-dessus de ce seuil, on considère que le
    texte n'a quasiment pas changé (variante mineure de formatage) et on
    traite comme un doublon classique. En dessous, on considère que c'est
    une vraie mise à jour de CV (nouveau contenu, ex: nouveaux projets).
    """
    unique: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    hash_map: Dict[str, Dict[str, Any]] = {}
    email_map: Dict[str, Dict[str, Any]] = {}
    phone_map: Dict[str, Dict[str, Any]] = {}
    embedding_cache: Dict[str, np.ndarray] = {}

    for entry in cvs:
        text = entry.get("text", "")
        filename = entry.get("filename", "<unknown>")
        data = entry.get("data")

        # ── Niveau 1 : hash exact ──────────────────────────────
        text_hash = _sha256(text)
        if text_hash in hash_map:
            report.append({
                "filename": filename,
                "duplicate_of": hash_map[text_hash]["filename"],
                "level": 1,
                "reason": "exact hash",
            })
            continue

        # ── Niveau 2 : même candidat (email / phone) ───────────
        email = None
        phone = None
        if isinstance(data, dict):
            email = (data.get("email") or "").strip().lower()
            phone = (data.get("telephone") or "").strip()
            phone = "".join(ch for ch in phone if ch.isdigit())

        matched_entry = None
        matched_reason = None
        if email and email in email_map:
            matched_entry = email_map[email]
            matched_reason = "email match"
        elif phone and phone in phone_map:
            matched_entry = phone_map[phone]
            matched_reason = "phone match"

        if matched_entry is not None:
            # On compare le texte brut pour voir si c'est une vraie mise à jour
            # ou juste une variante mineure du même contenu.
            cur_emb = embedding_model.encode(text)
            old_emb = embedding_model.encode(matched_entry.get("text", ""))
            sim = _cosine_similarity(cur_emb, old_emb)
            embedding_cache[filename] = cur_emb

            if sim >= update_similarity_threshold:
                # Texte quasi identique → vrai doublon, on exclut
                report.append({
                    "filename": filename,
                    "duplicate_of": matched_entry["filename"],
                    "level": 2,
                    "reason": f"{matched_reason} (contenu identique, similarité {sim:.2f})",
                })
                continue
            else:
                # Même candidat, mais contenu différent → CV mis à jour
                report.append({
                    "filename": filename,
                    "duplicate_of": matched_entry["filename"],
                    "level": 2,
                    "reason": f"{matched_reason} — CV mis à jour (similarité {sim:.2f}, nouveau contenu détecté)",
                    "action": "updated",
                })
                # On retire l'ancienne version de "unique" et on garde la nouvelle
                unique[:] = [e for e in unique if e["filename"] != matched_entry["filename"]]
                unique.append(entry)
                hash_map[text_hash] = entry
                if email:
                    email_map[email] = entry
                if phone:
                    phone_map[phone] = entry
                continue

        # ── Niveau 3 : similarité sémantique ───────────────────
        cur_emb = embedding_cache.get(filename) or embedding_model.encode(text)
        embedding_cache[filename] = cur_emb
        similar_found = False
        for kept in unique:
            kept_emb = embedding_cache.get(kept["filename"]).copy() if kept["filename"] in embedding_cache else embedding_model.encode(kept.get("text", ""))
            sim = _cosine_similarity(cur_emb, kept_emb)
            if sim >= similarity_threshold:
                report.append({
                    "filename": filename,
                    "duplicate_of": kept["filename"],
                    "level": 3,
                    "reason": f"semantic similarity {sim:.2f}",
                })
                similar_found = True
                break

        if similar_found:
            continue

        unique.append(entry)
        hash_map[text_hash] = entry
        if email:
            email_map[email] = entry
        if phone:
            phone_map[phone] = entry

    return unique, report


# ---------------------------------------------------------------------------
# Vérification interactive d'un seul CV (page "Ajouter un CV")
# ---------------------------------------------------------------------------

def check_single_cv_duplicates(
    text: str,
    data: Optional[Dict[str, Any]],
    file_hash: str,
    cache,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Vérifie un CV individuel contre le cache existant (3 niveaux).

    Paramètres
    ----------
    text : texte brut du CV
    data : données structurées extraites (contient email, telephone, nom…)
    file_hash : SHA-256 du fichier PDF source
    cache : instance de CVCache (doit exposer .get_by_hash() et .all_entries())
    similarity_threshold : seuil pour le niveau 3

    Retourne
    --------
    Dict avec les clés :
        - "level1_hit" : bool — True si hash exact trouvé
        - "level2_match" : dict | None — entrée du cache correspondante (email/phone)
        - "level3_top" : List[Tuple[float, str]] — top 3 CVs les plus similaires
                         (similarité, nom), triés par similarité décroissante
    """
    result: Dict[str, Any] = {
        "level1_hit": False,
        "level2_match": None,
        "level3_top": [],
    }

    # ── Niveau 1 — Hash exact ─────────────────────────────────
    cached_entry = cache.get_by_hash(file_hash) if hasattr(cache, "get_by_hash") else None
    if cached_entry:
        result["level1_hit"] = True

    # ── Niveau 2 — Email / téléphone ──────────────────────────
    email = (data.get("email") or "").strip().lower() if isinstance(data, dict) else ""
    telephone = (data.get("telephone") or "").strip() if isinstance(data, dict) else ""

    for entry in cache.all_entries():
        if entry.get("hash") == file_hash:
            continue
        entry_email = (entry.get("email") or "").strip().lower()
        entry_phone = (entry.get("phone") or "").strip()
        if email and entry_email == email:
            result["level2_match"] = entry
            break
        if telephone and entry_phone == telephone:
            result["level2_match"] = entry
            break

    # ── Niveau 3 — Similarité sémantique ──────────────────────
    emb_nouveau = embedding_model.encode(text)

    similarites = []
    for entry in cache.all_entries():
        if entry.get("hash") == file_hash:
            continue
        autre_texte = entry.get("text", "")
        if not autre_texte.strip():
            continue
        emb_autre = embedding_model.encode(autre_texte)
        sim = _cosine_similarity(emb_nouveau, emb_autre)
        nom_autre = (entry.get("data") or {}).get("nom", entry.get("source_path", "?"))
        similarites.append((sim, nom_autre))

    similarites.sort(key=lambda x: x[0], reverse=True)
    result["level3_top"] = similarites[:3]

    return result


def check_elasticsearch_status(
    es,
    file_hash: str,
    data: Optional[Dict[str, Any]],
    text: str,
    index: str = "cvs",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Vérifie si un CV existe déjà dans Elasticsearch (dashboard)."""
    result: Dict[str, Any] = {
        "already_indexed_by_hash": False,
        "es_identity_match": None,
        "es_top_similar": [],
        "indexed_doc_name": None,
    }
    if es is None or not hasattr(es, "search"):
        return result

    try:
        if hasattr(es, "exists") and es.exists(index=index, id=file_hash):
            result["already_indexed_by_hash"] = True
            doc = es.get(index=index, id=file_hash)["_source"]
            result["indexed_doc_name"] = doc.get("nom") or doc.get("filename") or file_hash[:12]
            return result
    except Exception:
        pass

    email = (data.get("email") or "").strip().lower() if isinstance(data, dict) else ""
    phone = _normalize_phone((data.get("telephone") or "") if isinstance(data, dict) else "")

    should_clauses = []
    if email:
        should_clauses.append({"term": {"email": email}})
    if phone:
        should_clauses.append({"term": {"telephone": phone}})

    if should_clauses:
        try:
            resp = es.search(
                index=index,
                size=1,
                query={"bool": {"should": should_clauses, "minimum_should_match": 1}},
                _source=["nom", "email", "telephone", "text", "filename"],
            )
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                result["es_identity_match"] = hits[0]
        except Exception:
            pass

    if not text.strip():
        return result

    try:
        total = es.count(index=index).get("count", 0)
        if total <= 0:
            return result

        resp = es.search(
            index=index,
            size=total,
            _source=["nom", "text"],
        )
        emb_new = embedding_model.encode(text)
        similarites = []
        for hit in resp.get("hits", {}).get("hits", []):
            if hit.get("_id") == file_hash:
                continue
            other_text = (hit.get("_source") or {}).get("text", "")
            if not other_text.strip():
                continue
            sim = _cosine_similarity(emb_new, embedding_model.encode(other_text))
            nom = (hit.get("_source") or {}).get("nom", "?")
            similarites.append((sim, nom, hit.get("_id")))

        similarites.sort(key=lambda x: x[0], reverse=True)
        result["es_top_similar"] = similarites[:3]
        top_sim = similarites[0][0] if similarites else 0.0
        result["es_high_similarity"] = top_sim >= similarity_threshold
    except Exception:
        pass

    return result


def assess_cv_indexability(
    text: str,
    data: Optional[Dict[str, Any]],
    file_hash: str,
    cache,
    es=None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    update_similarity_threshold: float = UPDATE_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Décide si un CV peut être indexé (non doublon + absent du dashboard)."""
    dup = check_single_cv_duplicates(
        text=text,
        data=data,
        file_hash=file_hash,
        cache=cache,
        similarity_threshold=similarity_threshold,
    )
    es_status = check_elasticsearch_status(
        es=es,
        file_hash=file_hash,
        data=data,
        text=text,
        similarity_threshold=similarity_threshold,
    )

    can_index = True
    status = "ok"
    reasons: List[str] = []

    if es_status.get("already_indexed_by_hash"):
        can_index = False
        status = "already_indexed"
        nom = es_status.get("indexed_doc_name") or "ce CV"
        reasons.append(f"Déjà présent dans le dashboard : « {nom} » est déjà indexé.")

    if dup.get("level2_match") and can_index:
        old_text = dup["level2_match"].get("text", "")
        nom = (dup["level2_match"].get("data") or {}).get("nom", "?")
        if old_text.strip():
            sim = _cosine_similarity(
                embedding_model.encode(text),
                embedding_model.encode(old_text),
            )
            if sim >= update_similarity_threshold:
                can_index = False
                status = "duplicate"
                reasons.append(
                    f"Doublon : même candidat « {nom} » (email/téléphone) avec contenu quasi identique."
                )
            elif sim >= similarity_threshold:
                can_index = False
                status = "duplicate"
                reasons.append(
                    f"Doublon probable : très similaire au CV existant de « {nom} » ({sim:.2f})."
                )

    if can_index:
        for sim, nom in dup.get("level3_top", []):
            if sim >= similarity_threshold:
                can_index = False
                status = "duplicate"
                reasons.append(
                    f"Doublon sémantique : contenu très similaire à « {nom} » ({sim:.2f})."
                )
                break

    if can_index and es_status.get("es_identity_match"):
        hit = es_status["es_identity_match"]
        src = hit.get("_source") or {}
        nom = src.get("nom") or src.get("filename") or "?"
        can_index = False
        status = "already_indexed"
        reasons.append(
            f"Déjà dans le dashboard : le candidat « {nom} » possède déjà un CV indexé "
            f"(même email ou téléphone)."
        )

    if can_index:
        for sim, nom, _doc_id in es_status.get("es_top_similar", []):
            if sim >= similarity_threshold:
                can_index = False
                status = "duplicate"
                reasons.append(
                    f"Doublon sémantique dans le dashboard : très similaire à « {nom} » ({sim:.2f})."
                )
                break

    return {
        **dup,
        **es_status,
        "can_index": can_index,
        "status": status,
        "reasons": reasons,
    }