#!/usr/bin/env python3
"""
setup.py — Script d'installation complet du projet cv-pipeline.
===============================================================
Lance UNE SEULE commande pour tout configurer :

    python setup.py

Ce script effectue dans l'ordre :
  0. Synchronise le cache — supprime les entrees dont le PDF n'existe plus
  1. Verifie les dependances (Elasticsearch, Kibana)
  2. Cree l'index Elasticsearch avec le bon mapping
  3. Traite et injecte les CVs :
       3a. Charge cvs_data.json (CVs deja extraits)
       3b. Scanne cvs/ et cvs_uploads/ pour decouvrir de nouveaux PDFs
       3c. Extrait les nouveaux PDFs via LLM (avec cache) → sauvegarde JSON + Excel
       3d. Injecte tous les CVs dans Elasticsearch
  4. Cree automatiquement le dashboard Kibana complet

Apres execution :
  - Streamlit  : streamlit run app.py
  - Kibana     : http://localhost:5601/app/dashboards
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import time
import requests

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with emoji in subprocesses)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


ROOT = os.path.dirname(os.path.abspath(__file__))
KIBANA_URL = os.getenv("KIBANA_URL", "http://localhost:5601")
ES_URL = os.getenv("ELASTIC_HOST", "http://localhost:9200")


class _SkipEntry(Exception):
    """Sentinel used to break out of the Level-3 dedup check cleanly."""


def print_step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 50)


def check_service(url: str, name: str, retries: int = 10, delay: int = 6) -> bool:
    print(f"  Attente de {name} ({url}) ...", end="", flush=True)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                print(" OK")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(delay)
    print(f" ECHEC")
    return False


def run(cmd: list[str], cwd: str = ROOT) -> int:
    """
    Lance un sous-processus avec encodage UTF-8 force.
    Evite les UnicodeEncodeError sur les terminaux Windows (cp1252).
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def main() -> None:
    # Force UTF-8 for this process's own stdout/stderr (Windows fix)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    total = 5
    print("\n" + "=" * 60)
    print("  CV Pipeline — Installation complete")
    print("=" * 60)

    # --- Etape 0 : Synchroniser le cache (supprimer les orphelins) ---
    print_step(0, total, "Synchronisation — suppression des entrees orphelines")
    try:
        from cv_sync import sync_orphans
        from elasticsearch import Elasticsearch
        _es = Elasticsearch(ES_URL)
        try:
            _es.cluster.health(timeout="3s")
            es_client = _es
        except Exception:
            es_client = None
            print("  Elasticsearch pas encore disponible — sync ES ignoree.")

        report = sync_orphans(es=es_client)
        total_orphans = report["removed_cache"] + report["removed_data"] + report["removed_es"]
        if total_orphans == 0:
            print("  Aucun orphelin detecte — tout est synchronise.")
        else:
            print(f"  Orphelins supprimes : {report['removed_cache']} cache | "
                  f"{report['removed_data']} data | "
                  f"{report['removed_es']} Elasticsearch")
            if report["orphaned_cache"]:
                for f in report["orphaned_cache"]:
                    print(f"    - {f}")
    except Exception as e:
        print(f"  [AVERTISSEMENT] Synchronisation ignoree : {e}")

    # --- Etape 1 : Verifier les services Docker ---
    print_step(1, total, "Verification des services (Elasticsearch + Kibana)")
    es_ok = check_service(f"{ES_URL}/_cluster/health", "Elasticsearch")
    kb_ok = check_service(f"{KIBANA_URL}/api/status", "Kibana")

    if not es_ok:
        print("\n  Elasticsearch n'est pas disponible.")
        print("  Lancez : docker compose up -d")
        print("  Puis relancez : python setup.py")
        sys.exit(1)

    if not kb_ok:
        print("\n  Kibana n'est pas encore pret (il peut mettre 30-60s a demarrer).")
        print("  Attendez quelques secondes puis relancez : python setup.py")
        sys.exit(1)

    print("  Elasticsearch et Kibana sont operationnels.")

    # --- Etape 2 : Creer l'index Elasticsearch ---
    print_step(2, total, "Creation de l'index Elasticsearch 'cvs'")
    ret = run([sys.executable, "create_index.py"])
    if ret != 0:
        print("  [AVERTISSEMENT] create_index.py a echoue ou l'index existait deja.")

    # --- Etape 3 : Traiter et injecter les CVs ---
    print_step(3, total, "Traitement et injection des CVs")

    # 3a — Charger les CVs deja extraits depuis cvs_data.json
    data_file = os.path.join(ROOT, "output", "cvs_data.json")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        print(f"  {len(existing_results)} CVs charges depuis cvs_data.json.")
    else:
        existing_results = []
        print("  cvs_data.json absent — sera cree.")

    # Build a set of already-extracted filenames for quick lookup
    already_extracted = {
        os.path.basename(
            (r.get("filename") or r.get("source_path") or "").replace("\\", "/")
        )
        for r in existing_results
    }

    # Build deduplication identifiers from already-loaded results
    # (email, phone, hash) — used to skip same-person different-filename PDFs
    def _seen_ids(results_list):
        emails, phones, hashes = set(), set(), set()
        for r in results_list:
            d = r.get("data") or {}
            em = (d.get("email") or "").strip().lower()
            ph = (d.get("telephone") or "").strip()
            h  = (r.get("hash") or "").strip()
            if em:  emails.add(em)
            if ph:  phones.add(ph)
            if h:   hashes.add(h)
        return emails, phones, hashes

    seen_emails, seen_phones, seen_hashes = _seen_ids(existing_results)

    # 3b — Scanner cvs/ et cvs_uploads/ pour decouvrir de nouveaux PDFs
    CV_FOLDERS = [
        os.path.join(ROOT, "cvs"),
        os.path.join(ROOT, "cvs_uploads"),
    ]
    new_pdfs = []
    for folder in CV_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".pdf") and fname not in already_extracted:
                new_pdfs.append(os.path.join(folder, fname))

    if new_pdfs:
        print(f"\n  {len(new_pdfs)} nouveau(x) PDF(s) detecte(s) — extraction en cours...")

        # 3c — Extraire les nouveaux PDFs via LLM (avec cache)
        try:
            from cv_cache import CVCache, file_sha256
            from cv_reader import read_cv_text
            from cv_extractor import extract_cv_data_auto, AVAILABLE_KEYS, FALLBACK_ORDER
            from cv_saver import save_to_json, save_to_excel
            from cv_deduplication import check_single_cv_duplicates, DEFAULT_SIMILARITY_THRESHOLD

            # Pick the first available LLM provider
            provider = next(
                (p for p in FALLBACK_ORDER if AVAILABLE_KEYS.get(p)),
                None,
            )
            if provider is None:
                print("  [AVERTISSEMENT] Aucune cle API LLM configuree dans .env.")
                print("  Les nouveaux PDFs ne seront pas extraits.")
                print("  Configurez GROQ_API_KEY, MISTRAL_API_KEY, GEMINI_API_KEY ou OPENROUTER_API_KEY.")
            else:
                print(f"  Fournisseur LLM : {provider}")
                cache = CVCache()
                newly_extracted = 0
                skipped_duplicates = 0

                for pdf_path in new_pdfs:
                    fname = os.path.basename(pdf_path)
                    print(f"  -> {fname} ...", end="", flush=True)
                    try:
                        file_hash = file_sha256(pdf_path)

                        # Level 1 — Skip if exact same file hash already processed
                        if file_hash in seen_hashes:
                            print(" DOUBLON (hash identique — meme fichier)")
                            skipped_duplicates += 1
                            continue

                        # Get data from cache (LLM call avoided) or extract fresh
                        cached = cache.get_by_hash(file_hash)
                        if cached:
                            data = cached.get("data") or {}
                            text = cached.get("text", "")
                            from_cache = True
                        else:
                            text = read_cv_text(pdf_path)
                            if len(text.strip()) < 20:
                                print(" IGNORE (texte vide)")
                                continue
                            data = extract_cv_data_auto(text, provider=provider)
                            from_cache = False

                        # Level 2 — Dedup by email / phone BEFORE saving to cache or data
                        # Same person, different filename -> skip entirely (not stored anywhere)
                        entry_email = (data.get("email") or "").strip().lower()
                        entry_phone = (data.get("telephone") or "").strip()
                        entry_nom   = data.get("nom", fname)

                        if entry_email and entry_email in seen_emails:
                            print(f" DOUBLON (meme email : {entry_email}) -- {entry_nom} ignore")
                            skipped_duplicates += 1
                            continue
                        if entry_phone and entry_phone in seen_phones:
                            print(f" DOUBLON (meme telephone : {entry_phone}) -- {entry_nom} ignore")
                            skipped_duplicates += 1
                            continue

                        # Level 3 — Semantic similarity (same person, no email/phone match)
                        # Uses embedding cosine similarity against all already-processed texts
                        # Only run if there are existing CVs to compare against
                        if text.strip() and existing_results:
                            try:
                                dup = check_single_cv_duplicates(
                                    text=text,
                                    data=data,
                                    file_hash=file_hash,
                                    cache=cache,
                                    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                                )
                                for sim, nom_sim in dup.get("level3_top", []):
                                    if sim >= DEFAULT_SIMILARITY_THRESHOLD:
                                        print(f" DOUBLON semantique : similaire a '{nom_sim}' ({sim:.2f}) -- ignore")
                                        skipped_duplicates += 1
                                        raise _SkipEntry()
                            except _SkipEntry:
                                continue
                            except Exception as e_l3:
                                pass  # Level 3 non-bloquant si embedding indisponible

                        # All 3 levels passed — unique person confirmed
                        # Now safe to write to cache (only if not already cached)
                        if not from_cache:
                            cache.insert({
                                "hash": file_hash,
                                "email": entry_email,
                                "phone": entry_phone,
                                "data": data,
                                "text": text,
                                "source_path": pdf_path,
                            })
                            print(" OK")
                        else:
                            print(" (cache)")

                        entry = {
                            "filename": fname,
                            "text": text,
                            "data": data,
                            "provider": "cache" if from_cache else provider,
                            "hash": file_hash,
                        }

                        # Register identifiers to prevent later entries from matching
                        if entry_email: seen_emails.add(entry_email)
                        if entry_phone: seen_phones.add(entry_phone)
                        seen_hashes.add(file_hash)

                        # Add to results (replace if same filename already present)
                        existing_results = [
                            r for r in existing_results
                            if os.path.basename(
                                (r.get("filename") or r.get("source_path") or "").replace("\\", "/")
                            ) != fname
                        ]
                        existing_results.append(entry)
                        newly_extracted += 1


                    except Exception as e:
                        print(f" ERREUR ({e})")

                if skipped_duplicates:
                    print(f"  {skipped_duplicates} doublon(s) ignore(s) (meme email/telephone/hash).") 

                # Save updated cvs_data.json + Excel
                if newly_extracted > 0:
                    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
                    save_to_json(existing_results, data_file)
                    save_to_excel(existing_results)
                    print(f"  {newly_extracted} nouveau(x) CV(s) extraits et sauvegardes.")

        except Exception as e:
            print(f"  [ERREUR] Echec de l extraction des nouveaux PDFs : {e}")
    else:
        print("  Aucun nouveau PDF detecte dans cvs/ et cvs_uploads/.")

    # 3d — Injecter tous les CVs dans Elasticsearch
    if os.path.exists(data_file):
        print("\n  Injection de tous les CVs dans Elasticsearch...")
        ret = run([sys.executable, "cv_injector.py"])
        if ret != 0:
            print("  [AVERTISSEMENT] cv_injector.py a signale une erreur.")
        else:
            print("  CVs injectes avec succes.")
    else:
        print(f"  Fichier {data_file} absent — injection ignoree.")
        print("  Pour traiter vos CVs manuellement : python cv_extractor.py && python cv_saver.py")

    # --- Etape 4 : Creer le dashboard Kibana ---
    print_step(4, total, "Creation automatique du dashboard Kibana")
    setup_kibana = os.path.join(ROOT, "scripts", "setup_kibana.py")
    ret = run([sys.executable, setup_kibana])
    if ret != 0:
        print("  [AVERTISSEMENT] Le dashboard Kibana n'a pas pu etre cree.")

    # --- Fin ---
    print("\n" + "=" * 60)
    print("  Installation terminee !")
    print("=" * 60)
    print(f"\n  Streamlit  : streamlit run app.py")
    print(f"  Kibana     : {KIBANA_URL}/app/dashboards")
    print(f"  Elasticsearch : {ES_URL}")
    print()


if __name__ == "__main__":
    main()
