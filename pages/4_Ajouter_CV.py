"""
pages/4_Ajouter_CV.py
Page Streamlit : upload d'un CV PDF, extraction via LLM, preview, puis
sauvegarde (JSON/Excel) + indexation dans Elasticsearch.

Règles d'indexation :
- Doublon détecté (hash, email/tél., similarité) → indexation bloquée
- CV déjà présent dans le dashboard (Elasticsearch) → indexation bloquée
- Nouveau CV unique et absent du dashboard → indexation autorisée
"""

import os
import json
from datetime import datetime, timezone

import streamlit as st

from es_client import get_es_client
from cv_reader import read_cv_text
from cv_extractor import extract_cv_data_auto, AVAILABLE_KEYS, FALLBACK_ORDER
from cv_cache import CVCache, file_sha256
from cv_saver import save_to_json, save_to_excel
from cv_deduplication import assess_cv_indexability, DEFAULT_SIMILARITY_THRESHOLD

st.set_page_config(page_title="Ajouter CV", layout="wide")
st.title("📝 Ajouter un CV")

UPLOAD_DIR = "cvs_uploads"
JSON_PATH = "output/cvs_data.json"
INDEX_NAME = "cvs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

es = get_es_client()
cache = CVCache()


def reset_upload_state():
    st.session_state.extraction_result = None
    st.session_state.extraction_hash = None
    st.session_state.index_success = False


providers_dispo = [p for p in FALLBACK_ORDER if AVAILABLE_KEYS.get(p)]
if not providers_dispo:
    st.error("❌ Aucune clé API LLM configurée dans .env (Groq, Mistral, Gemini, OpenRouter).")
    st.stop()

provider = st.selectbox("Fournisseur LLM", providers_dispo, index=0)

uploaded_file = st.file_uploader("Dépose un CV (PDF)", type="pdf", key="cv_uploader")

if uploaded_file is None:
    if st.session_state.get("index_success"):
        st.success("✅ Le dernier CV a été indexé. Déposez un nouveau PDF pour continuer.")
    else:
        st.info("👆 Dépose un fichier PDF pour commencer.")
    st.stop()

temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

file_hash = file_sha256(temp_path)

if "extraction_result" not in st.session_state or st.session_state.get("extraction_hash") != file_hash:
    st.session_state.extraction_result = None
    st.session_state.extraction_hash = file_hash
    st.session_state.index_success = False

cached_entry = cache.get_by_hash(file_hash)

if cached_entry and st.session_state.extraction_result is None:
    st.warning(
        "ℹ️ Ce CV a déjà été extrait auparavant (même contenu, hash identique). "
        "Résultat rechargé depuis le cache — aucun nouvel appel LLM effectué."
    )
    st.session_state.extraction_result = {
        "filename": uploaded_file.name,
        "text": cached_entry.get("text", ""),
        "data": cached_entry.get("data"),
    }

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
                cache.insert({
                    "hash": file_hash,
                    "email": (data.get("email") or "").strip().lower(),
                    "phone": (data.get("telephone") or "").strip(),
                    "data": data,
                    "text": text,
                    "source_path": temp_path,
                })
                st.success("✅ Extraction terminée.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Échec de l'extraction : {e}")
                st.stop()
    else:
        st.stop()

result = st.session_state.extraction_result
data = result["data"]

st.subheader("Aperçu de l'extraction")

col1, col2, col3 = st.columns(3)
col1.metric("Nom", data.get("nom", "—"))
col2.metric("Catégorie principale", data.get("categorie_principale", "—"))
col3.metric("Score qualité", f"{data.get('score_qualite_globale', '—')}/100")

with st.expander("📄 Voir le JSON complet extrait"):
    st.json(data)

st.subheader("🔍 Vérification avant indexation")

assessment = assess_cv_indexability(
    text=result["text"],
    data=data,
    file_hash=file_hash,
    cache=cache,
    es=es,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
)

if assessment["level1_hit"]:
    st.error("🚫 Niveau 1 (hash) : contenu identique à un CV déjà traité — doublon exact.")
else:
    st.success("✅ Niveau 1 (hash) : nouveau contenu.")

level2_match = assessment.get("level2_match")
if level2_match:
    nom_existant = (level2_match.get("data") or {}).get("nom", "?")
    st.warning(f"⚠️ Niveau 2 (email/téléphone) : correspondance avec « {nom_existant} ».")
else:
    st.caption("Niveau 2 (email/téléphone) : aucun candidat correspondant dans le cache.")

for sim, nom_autre in assessment.get("level3_top", []):
    if sim >= DEFAULT_SIMILARITY_THRESHOLD:
        st.warning(f"⚠️ Niveau 3 : similarité {sim:.2f} avec « {nom_autre} » (seuil {DEFAULT_SIMILARITY_THRESHOLD}).")
    else:
        st.caption(f"Niveau 3 : « {nom_autre} » — similarité {sim:.2f}")

if assessment.get("already_indexed_by_hash"):
    st.error("🚫 Dashboard : ce fichier est déjà indexé dans Elasticsearch.")
elif assessment.get("es_identity_match"):
    src = assessment["es_identity_match"].get("_source") or {}
    st.error(
        f"🚫 Dashboard : « {src.get('nom', '?')} » est déjà présent "
        f"(même email ou téléphone)."
    )
else:
    st.success("✅ Dashboard : ce CV n'est pas encore indexé.")

st.divider()

if assessment["can_index"]:
    st.success("✅ Ce CV peut être ajouté : il n'est ni un doublon ni déjà présent dans le dashboard.")
else:
    st.error("🚫 Indexation bloquée :")
    for reason in assessment.get("reasons", []):
        st.markdown(f"- {reason}")

if assessment["can_index"]:
    if st.button("✅ Valider et indexer", type="primary"):
        with st.spinner("Sauvegarde et indexation en cours..."):
            try:
                recheck = assess_cv_indexability(
                    text=result["text"],
                    data=data,
                    file_hash=file_hash,
                    cache=cache,
                    es=es,
                    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                )
                if not recheck["can_index"]:
                    st.error("🚫 Indexation refusée — le CV est devenu un doublon ou existe déjà dans le dashboard.")
                    for reason in recheck.get("reasons", []):
                        st.markdown(f"- {reason}")
                    st.stop()

                if os.path.exists(JSON_PATH):
                    with open(JSON_PATH, "r", encoding="utf-8") as f:
                        all_results = json.load(f)
                else:
                    all_results = []

                all_results = [r for r in all_results if r.get("filename") != result["filename"]]
                all_results.append({
                    "filename": result["filename"],
                    "text": result["text"],
                    "data": data,
                    "provider": provider,
                })

                save_to_json(all_results, JSON_PATH)
                save_to_excel(all_results)

                doc = dict(data)
                doc["filename"] = result["filename"]
                doc["text"] = result["text"]
                doc["indexed_at"] = datetime.now(timezone.utc).isoformat()

                es.index(index=INDEX_NAME, id=file_hash, document=doc)

                st.session_state.index_success = True
                st.success(f"🎉 CV indexé avec succès dans '{INDEX_NAME}' (ID : {file_hash[:12]}...)")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Échec de la sauvegarde/indexation : {e}")
else:
    st.button("✅ Valider et indexer", type="primary", disabled=True)
    st.caption("Corrigez le problème (autre PDF ou suppression du doublon existant) pour continuer.")

if st.session_state.get("index_success"):
    if st.button("➕ Ajouter un autre CV"):
        reset_upload_state()
        st.rerun()
