"""
pages/4_Ajouter_CV.py
Page Streamlit : upload d'un CV PDF, extraction via LLM, preview, puis
sauvegarde (JSON/Excel) + indexation dans Elasticsearch.
"""

import os
import json
import streamlit as st

from es_client import get_es_client
from cv_reader import read_cv_text
from cv_extractor import extract_cv_data_auto, AVAILABLE_KEYS, FALLBACK_ORDER
from cv_cache import CVCache, file_sha256
from cv_saver import save_to_json, save_to_excel
from cv_deduplication import embedding_model, _cosine_similarity, DEFAULT_SIMILARITY_THRESHOLD

st.set_page_config(page_title="Ajouter CV", layout="wide")
st.title("📝 Ajouter un CV")

UPLOAD_DIR = "cvs_uploads"
JSON_PATH = "output/cvs_data.json"
INDEX_NAME = "cvs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

es = get_es_client()
cache = CVCache()


# ─────────────────────────────────────────────────────────────
# Choix du provider LLM
# ─────────────────────────────────────────────────────────────
providers_dispo = [p for p in FALLBACK_ORDER if AVAILABLE_KEYS.get(p)]
if not providers_dispo:
    st.error("❌ Aucune clé API LLM configurée dans .env (Groq, Mistral, Gemini, OpenRouter).")
    st.stop()

provider = st.selectbox("Fournisseur LLM", providers_dispo, index=0)

# ─────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Dépose un CV (PDF)", type="pdf")

if uploaded_file is None:
    st.info("👆 Dépose un fichier PDF pour commencer.")
    st.stop()

# Écriture temporaire sur disque (cv_reader attend un chemin fichier)
temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

file_hash = file_sha256(temp_path)

# Init de l'état de session pour garder l'extraction entre les reruns Streamlit
if "extraction_result" not in st.session_state or st.session_state.get("extraction_hash") != file_hash:
    st.session_state.extraction_result = None
    st.session_state.extraction_hash = file_hash

# ─────────────────────────────────────────────────────────────
# Cache : ce CV a-t-il déjà été traité ?
# ─────────────────────────────────────────────────────────────
cached_entry = cache.get_by_hash(file_hash)

if cached_entry and st.session_state.extraction_result is None:
    st.warning("ℹ️ Ce CV a déjà été extrait auparavant (même contenu, hash identique). "
               "Résultat rechargé depuis le cache — aucun nouvel appel LLM effectué.")
    st.session_state.extraction_result = {
        "filename": uploaded_file.name,
        "text": cached_entry.get("text", ""),
        "data": cached_entry.get("data"),
    }

# ─────────────────────────────────────────────────────────────
# Extraction (si pas déjà en cache / pas déjà lancée)
# ─────────────────────────────────────────────────────────────
if st.session_state.extraction_result is None:
    if st.button("🚀 Lancer l'extraction"):
        with st.spinner(f"Lecture du PDF et extraction via {provider}..."):
            try:
                text = read_cv_text(temp_path)
                if len(text.strip()) < 20:
                    st.error("❌ Le texte extrait du PDF est quasi vide — vérifie que le fichier n'est pas corrompu.")
                    st.stop()

                data = extract_cv_data_auto(text, provider=provider)

                st.session_state.extraction_result = {
                    "filename": uploaded_file.name,
                    "text": text,
                    "data": data,
                }
                # On enregistre tout de suite dans le cache pour éviter de repayer
                # l'appel LLM si l'utilisateur revient sur ce même fichier.
                cache.insert({
                    "hash": file_hash,
                    "email": (data.get("email") or "").strip().lower(),
                    "phone": (data.get("telephone") or "").strip(),
                    "data": data,
                    "text": text,
                    "source_path": temp_path,
                })
                st.success("✅ Extraction terminée.")
            except Exception as e:
                st.error(f"❌ Échec de l'extraction : {e}")
                st.stop()
    else:
        st.stop()

result = st.session_state.extraction_result
data = result["data"]

# ─────────────────────────────────────────────────────────────
# Preview avant validation
# ─────────────────────────────────────────────────────────────
st.subheader("Aperçu de l'extraction")

col1, col2, col3 = st.columns(3)
col1.metric("Nom", data.get("nom", "—"))
col2.metric("Catégorie principale", data.get("categorie_principale", "—"))
col3.metric("Score qualité", f"{data.get('score_qualite_globale', '—')}/100")

with st.expander("📄 Voir le JSON complet extrait"):
    st.json(data)

st.subheader("🔍 Vérification de doublon")

# ── Niveau 1 — Hash exact (déjà fait AVANT l'extraction) ─────
# Si le hash était déjà en cache, on sait déjà que c'est un doublon
# exact : pas besoin d'aller plus loin (niveaux 2/3 inutiles).
if cached_entry:
    st.info("✅ Niveau 1 (hash) : contenu identique à un CV déjà traité. "
            "C'est un doublon exact — niveaux 2/3 non nécessaires.")
else:
    st.caption("Niveau 1 (hash) : nouveau contenu, jamais vu.")

    # ── Niveau 2 — Email/téléphone (même candidat, CV mis à jour) ──
    email = (data.get("email") or "").strip().lower()
    telephone = (data.get("telephone") or "").strip()
    doublon_email = None

    for entry in cache.all_entries():
        if entry["hash"] == file_hash:
            continue
        entry_email = (entry.get("email") or "").strip().lower()
        entry_phone = (entry.get("phone") or "").strip()
        if email and entry_email == email:
            doublon_email = entry
            break
        if telephone and entry_phone == telephone:
            doublon_email = entry
            break

    if doublon_email:
        nom_existant = (doublon_email.get("data") or {}).get("nom", "?")
        st.warning(f"⚠️ Niveau 2 (email/téléphone) : même candidat déjà présent "
                   f"(« {nom_existant} »). Probablement une nouvelle version de son CV.")
    else:
        st.caption("Niveau 2 (email/téléphone) : aucun candidat correspondant.")

    # ── Niveau 3 — Similarité sémantique sur le TEXTE BRUT ──────
    # (pas embedding_cv, qui est enrichi pour le RAG — ici on veut
    # comparer le texte du CV tel quel, comme fait cv_deduplication.py)
    text_brut = result["text"]
    emb_nouveau = embedding_model.encode(text_brut)

    similarites = []
    for entry in cache.all_entries():
        if entry["hash"] == file_hash:
            continue
        autre_texte = entry.get("text", "")
        if not autre_texte.strip():
            continue
        emb_autre = embedding_model.encode(autre_texte)
        sim = _cosine_similarity(emb_nouveau, emb_autre)
        nom_autre = (entry.get("data") or {}).get("nom", entry.get("source_path", "?"))
        similarites.append((sim, nom_autre))

    # On ne garde que le top 3, pas tout comparer à tout dans l'affichage
    similarites.sort(key=lambda x: x[0], reverse=True)
    top3 = similarites[:3]

    if top3:
        st.caption("Niveau 3 (similarité sémantique, top 3) :")
        for sim, nom_autre in top3:
            if sim >= DEFAULT_SIMILARITY_THRESHOLD:
                st.warning(f"⚠️ « {nom_autre} » — similarité {sim:.2f} "
                           f"(≥ seuil {DEFAULT_SIMILARITY_THRESHOLD}) → doublon probable")
            else:
                st.caption(f"« {nom_autre} » — similarité {sim:.2f}")
    else:
        st.caption("Niveau 3 (similarité) : aucun autre CV en cache pour comparer.")

# ─────────────────────────────────────────────────────────────
# Validation → sauvegarde JSON/Excel + indexation ES
# ─────────────────────────────────────────────────────────────
if st.button("✅ Valider et indexer", type="primary"):
    with st.spinner("Sauvegarde et indexation en cours..."):
        try:
            # 1. Append au JSON global (charge l'existant, ajoute, resauvegarde)
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            else:
                all_results = []

            # Évite les doublons dans le fichier JSON si on re-valide le même hash
            all_results = [r for r in all_results if r.get("filename") != result["filename"]]
            all_results.append({
                "filename": result["filename"],
                "text": result["text"],
                "data": data,
                "provider": provider,
            })

            save_to_json(all_results, JSON_PATH)
            save_to_excel(all_results)

            # 2. Indexation directe dans Elasticsearch (ID stable = hash du fichier,
            #    donc un ré-upload du même CV met juste à jour le doc au lieu d'en créer un doublon)
            doc = dict(data)
            doc["filename"] = result["filename"]
            doc["text"] = result["text"]
            # embedding_cv est déjà calculé par extract_cv_data(), pas besoin de le refaire

            es.index(index=INDEX_NAME, id=file_hash, document=doc)

            st.success(f"🎉 CV indexé avec succès dans '{INDEX_NAME}' (ID : {file_hash[:12]}...)")
            st.balloons()

            # Reset pour permettre l'ajout d'un nouveau CV
            if st.button("➕ Ajouter un autre CV"):
                st.session_state.extraction_result = None
                st.rerun()

        except Exception as e:
            st.error(f"❌ Échec de la sauvegarde/indexation : {e}")