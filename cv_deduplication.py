# cv_deduplication.py
"""
Utility module for three‑level CV duplicate detection.

Levels:
1️⃣ Exact duplicate – SHA‑256 hash of the raw CV text.
2️⃣ Same candidate – matching email **or** phone number.
3️⃣ Potential duplicate – high semantic similarity between the full texts.

The module provides a single public function ``detect_duplicates`` which
returns a list of unique CV entries (to be passed further down the pipeline)
and a report of detected duplicates.
"""

import hashlib
import os
import json
from typing import List, Tuple, Dict, Any
import numpy as np

# Re‑use the embedding model from the extractor package.
# The model is loaded when ``cv_extractor`` is imported, so importing it
# does not incur an additional download.
from cv_extractor import embedding_model

# Default similarity threshold for level 3 detection. Can be overridden
# via the environment variable ``DUPLICATE_SIMILARITY_THRESHOLD``.
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get(
    "DUPLICATE_SIMILARITY_THRESHOLD", "0.90"
))


def _sha256(text: str) -> str:
    """Return the SHA‑256 hash (hex) of *text*.

    The function works on the exact string that was read from the CV file –
    no normalisation is applied, because an exact match of the raw text
    guarantees a true duplicate.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1‑D numpy arrays."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def detect_duplicates(
    cvs: List[Dict[str, Any]],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Detect duplicates in *cvs*.

    Parameters
    ----------
    cvs:
        List of dictionaries produced by ``extract_all_cvs``. Each element
        must contain at least ``filename`` and ``text`` keys. ``data`` may be
        ``None`` for failed extractions.
    similarity_threshold:
        Cosine similarity cutoff for level 3 detection.

    Returns
    -------
    unique_cvs, duplicate_report
        ``unique_cvs`` is the list that should continue through the pipeline.
        ``duplicate_report`` contains entries of the form::

            {
                "filename": "CV_John.pdf",
                "duplicate_of": "CV_John_v1.pdf",
                "level": 2,
                "reason": "email match"
            }
    """
    unique: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    # Helper maps for fast lookup
    hash_map: Dict[str, Dict[str, Any]] = {}
    email_map: Dict[str, Dict[str, Any]] = {}
    phone_map: Dict[str, Dict[str, Any]] = {}
    # Cache embeddings for similarity checks
    embedding_cache: Dict[str, np.ndarray] = {}

    for entry in cvs:
        text = entry.get("text", "")
        filename = entry.get("filename", "<unknown>")
        data = entry.get("data")
        # ---------------------------------------------------------------
        # Level 1 – exact text hash
        # ---------------------------------------------------------------
        text_hash = _sha256(text)
        if text_hash in hash_map:
            report.append({
                "filename": filename,
                "duplicate_of": hash_map[text_hash]["filename"],
                "level": 1,
                "reason": "exact hash",
            })
            continue
        # ---------------------------------------------------------------
        # Level 2 – same candidate (email / phone)
        # ---------------------------------------------------------------
        email = None
        phone = None
        if isinstance(data, dict):
            email = (data.get("email") or "").strip().lower()
            phone = (data.get("telephone") or "").strip()
            phone = "".join(ch for ch in phone if ch.isdigit())
        matched = False
        if email:
            if email in email_map:
                report.append({
                    "filename": filename,
                    "duplicate_of": email_map[email]["filename"],
                    "level": 2,
                    "reason": "email match",
                })
                matched = True
        if not matched and phone:
            if phone and phone in phone_map:
                report.append({
                    "filename": filename,
                    "duplicate_of": phone_map[phone]["filename"],
                    "level": 2,
                    "reason": "phone match",
                })
                matched = True
        if matched:
            continue
        # ---------------------------------------------------------------
        # Level 3 – high semantic similarity
        # ---------------------------------------------------------------
        cur_emb = embedding_model.encode(text)
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
        # Register as unique if not discarded by any previous level
        unique.append(entry)
        hash_map[text_hash] = entry
        if email:
            email_map[email] = entry
        if phone:
            phone_map[phone] = entry

    return unique, report


def save_duplicate_report(report: List[Dict[str, Any]], path: str = "output/duplicates.json") -> None:
    """Save *report* as pretty‑printed JSON.

    The function creates the ``output`` directory if necessary.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ Duplicate report saved: {path}")
