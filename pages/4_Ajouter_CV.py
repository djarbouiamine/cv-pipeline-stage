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
from cv_removal import remove_cv

st.set_page_config(page_title="📤 Ajouter CV", layout="wide", page_icon="📤")

import sys as _sys3, os as _os3
_sys3.path.insert(0, _os3.path.join(_os3.path.dirname(__file__), ".."))
from theme import inject_theme, hero
inject_theme()
hero("📤", "Ajouter un CV", "Upload PDF • Extraction LLM • Détection doublons • Indexation Elasticsearch", badge="📂 cvs index")


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
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1


providers_dispo = [p for p in FALLBACK_ORDER if AVAILABLE_KEYS.get(p)]
if not providers_dispo:
    st.error("❌ Aucune clé API LLM configurée dans .env (Groq, Mistral, Gemini, OpenRouter).")
    st.stop()

provider = st.selectbox("Fournisseur LLM", providers_dispo, index=0)

uploaded_file = st.file_uploader(
    "Dépose un CV (PDF)",
    type="pdf",
    key=f"cv_uploader_{st.session_state.get('uploader_nonce', 0)}",
)

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

# ── Apercu extraction ──────────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
border-radius:14px;padding:1.1rem 1.5rem;margin:0.8rem 0">
  <p style="color:rgba(255,255,255,0.5);font-size:0.72rem;font-weight:600;
  letter-spacing:.8px;text-transform:uppercase;margin:0 0 0.6rem">Aperçu de l'extraction</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("NOM", data.get("nom", "—"))
col2.metric("CATÉGORIE PRINCIPALE", data.get("categorie_principale", "—"))
score_val = data.get('score_qualite_globale', '')
col3.metric("SCORE QUALITÉ", f"{score_val}/100" if score_val != '' else "—")

# Toggle JSON (sans st.expander)
if "show_json_4" not in st.session_state:
    st.session_state.show_json_4 = False
col_j, _ = st.columns([1, 6])
with col_j:
    lbl = "📄 Voir JSON" if not st.session_state.show_json_4 else "📄 Masquer JSON"
    if st.button(lbl, key="toggle_json_cv4"):
        st.session_state.show_json_4 = not st.session_state.show_json_4
if st.session_state.show_json_4:
    st.json(data)

st.markdown("""
<div style="background:rgba(79,172,254,0.07);border-left:4px solid #4facfe;
border-radius:0 10px 10px 0;padding:0.8rem 1.2rem;margin:0.8rem 0">
  <span style="color:#4facfe;font-weight:700;font-size:0.9rem">🔍 Vérification avant indexation</span>
</div>
""", unsafe_allow_html=True)


assessment = assess_cv_indexability(
    text=result["text"],
    data=data,
    file_hash=file_hash,
    cache=cache,
    es=es,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
)

if assessment["level1_hit"]:
    if assessment.get("already_indexed_by_hash"):
        st.success("✅ Niveau 1 (hash) : fichier déjà indexé dans le dashboard.")
    else:
        st.info(
            "ℹ️ Niveau 1 (hash) : déjà extrait (cache) — "
            "vous pouvez l'indexer dans le dashboard."
        )
else:
    st.success("✅ Niveau 1 (hash) : nouveau contenu.")

level2_match = assessment.get("level2_match")
if level2_match:
    nom_existant = (level2_match.get("data") or {}).get("nom", "?")
    st.warning(f"⚠️ Niveau 2 (email/téléphone) : correspondance avec « {nom_existant} ».")
else:
    st.caption("Niveau 2 (email/téléphone) : aucun candidat correspondant dans le cache.")

level3_top = assessment.get("level3_top", [])
if level3_top:
    for sim, nom_autre in level3_top:
        if sim >= DEFAULT_SIMILARITY_THRESHOLD:
            st.warning(
                f"⚠️ Niveau 3 (similarité sémantique) : {sim:.2f} avec « {nom_autre} » "
                f"(seuil {DEFAULT_SIMILARITY_THRESHOLD})."
            )
        else:
            st.caption(f"Niveau 3 : « {nom_autre} » — similarité {sim:.2f}")
else:
    st.caption("Niveau 3 (similarité sémantique) : aucun CV similaire trouvé dans le cache.")

if assessment.get("already_indexed_by_hash"):
    st.success("✅ Dashboard : ce CV est déjà indexé.")
elif assessment.get("es_identity_match"):
    src = assessment["es_identity_match"].get("_source") or {}
    st.error(
        f"🚫 Dashboard : « {src.get('nom', '?')} » est déjà présent "
        f"(même email ou téléphone)."
    )
else:
    st.success("✅ Dashboard : ce CV n'est pas encore indexé.")

st.divider()

already_in_dashboard = assessment.get("already_indexed_by_hash")

if already_in_dashboard:
    st.session_state.index_success = True
    st.success(
        f"✅ **{data.get('nom', 'Ce CV')}** est déjà présent dans le dashboard. "
        "Aucune action nécessaire."
    )
    st.info(
        "Ce PDF a déjà été indexé (même fichier). Pour le réindexer avec de nouvelles données, "
        "supprimez-le d'abord puis uploadez-le à nouveau."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Ajouter un autre CV", width='stretch'):
            reset_upload_state()
            st.rerun()
    with col_b:
        confirm_replace = st.checkbox("Je veux supprimer puis réindexer", key="confirm_replace_cv")
        if st.button(
            "🗑️ Supprimer du dashboard",
            disabled=not confirm_replace,
            width='stretch',
            key="replace_cv_btn",
        ):
            source = {
                "filename": result["filename"],
                "email": data.get("email"),
                "nom": data.get("nom"),
            }
            report = remove_cv(es, cache, file_hash, source)
            reset_upload_state()
            if report.get("success"):
                st.success("✅ CV supprimé. Uploadez à nouveau le PDF pour réindexer.")
            else:
                st.error("❌ Suppression impossible. Vérifiez qu'Elasticsearch est démarré.")
            st.rerun()
    st.stop()

if not assessment["can_index"]:
    st.session_state.index_success = False

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
                st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
                st.success(f"🎉 CV indexé avec succès dans '{INDEX_NAME}' (ID : {file_hash[:12]}...)")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Échec de la sauvegarde/indexation : {e}")
else:
    st.button("✅ Valider et indexer", type="primary", disabled=True)
    st.caption("Corrigez le problème (autre PDF ou suppression du doublon existant) pour continuer.")

    st.divider()
    st.subheader("🗑️ Supprimer les données existantes")
    st.warning(
        "Supprime ce CV du cache, des uploads et du dashboard (si présent), "
        "puis relance l'extraction depuis le début."
    )
    confirm_purge = st.checkbox("Je confirme la suppression", key="confirm_purge_cv")
    if st.button(
        "🗑️ Supprimer et recommencer",
        disabled=not confirm_purge,
        key="purge_cv_btn",
    ):
        source = {
            "filename": result["filename"],
            "email": data.get("email"),
            "nom": data.get("nom"),
        }
        report = remove_cv(es, cache, file_hash, source)
        reset_upload_state()
        if report.get("success"):
            parts = []
            if report.get("elasticsearch"):
                parts.append(f"dashboard ({report.get('elasticsearch_count', 1)})")
            if report.get("cache"):
                parts.append(f"cache ({report.get('cache_count', 1)})")
            if report.get("pdf"):
                parts.append("PDF cvs/uploads")
            if report.get("output"):
                parts.append("JSON/Excel")
            st.success(f"✅ Supprimé : {', '.join(parts)}. Déposez à nouveau le PDF.")
        else:
            st.error("❌ Aucune donnée n'a pu être supprimée. Vérifiez qu'Elasticsearch est démarré.")
        st.rerun()

if st.session_state.get("index_success") and assessment["can_index"]:
    if st.button("➕ Ajouter un autre CV"):
        reset_upload_state()
        st.rerun()

# ── Lien vers la page Gestion CVs ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.info(
    "🗂️ **Analyser un CV existant ou supprimer un CV complètement ?** "
    "→ Rendez-vous sur la page **Gestion CVs** dans le menu de gauche."
)

# ── Synchronisation du cache ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()


with st.expander("🔄 Synchroniser le cache — supprimer les CVs effacés du disque", expanded=False):
    st.markdown("""
<div style="background:rgba(255,165,0,0.07);border-left:4px solid #f6d365;
border-radius:0 10px 10px 0;padding:0.8rem 1.2rem;margin-bottom:1rem">
  <span style="color:#f6d365;font-weight:700;font-size:0.9rem">⚠️ Orphelins détectés ?</span><br>
  <span style="color:rgba(255,255,255,0.6);font-size:0.82rem">
  Si vous avez supprimé manuellement un PDF du dossier <code>cvs/</code> ou
  <code>cvs_uploads/</code>, son extraction reste dans le cache et les fichiers JSON.
  Cliquez sur <strong>Analyser</strong> pour voir les orphelins, puis <strong>Nettoyer</strong>
  pour les supprimer.
  </span>
</div>
""", unsafe_allow_html=True)

    try:
        from cv_sync import sync_orphans, _all_pdf_filenames

        # Dry-run preview
        col_analyze, col_clean = st.columns(2)
        with col_analyze:
            if st.button("🔍 Analyser (aperçu)", width='stretch', key="btn_sync_analyze"):
                with st.spinner("Analyse en cours..."):
                    preview = sync_orphans(es=es, dry_run=True)
                st.session_state["sync_preview"] = preview

        with col_clean:
            has_orphans = bool(
                st.session_state.get("sync_preview", {}).get("orphaned_cache")
                or st.session_state.get("sync_preview", {}).get("orphaned_data")
                or st.session_state.get("sync_preview", {}).get("orphaned_es")
            )
            if st.button(
                "🧹 Nettoyer les orphelins",
                width='stretch',
                key="btn_sync_clean",
                disabled=not has_orphans,
                type="primary" if has_orphans else "secondary",
            ):
                with st.spinner("Nettoyage en cours..."):
                    result_sync = sync_orphans(es=es, dry_run=False)
                st.session_state["sync_preview"] = None
                total_removed = (
                    result_sync["removed_cache"]
                    + result_sync["removed_data"]
                    + result_sync["removed_es"]
                )
                if total_removed == 0:
                    st.info("ℹ️ Aucun orphelin trouvé — le cache est déjà synchronisé.")
                else:
                    st.success(
                        f"✅ Supprimés : **{result_sync['removed_cache']}** cache · "
                        f"**{result_sync['removed_data']}** data · "
                        f"**{result_sync['removed_es']}** Elasticsearch"
                    )
                    if result_sync["es_skipped"]:
                        st.warning("⚠️ Elasticsearch non disponible — suppression ES ignorée.")
                st.rerun()

        # Show preview results
        preview = st.session_state.get("sync_preview")
        if preview is not None:
            n_pdfs = len(preview["existing_pdfs"])
            n_orphan_cache = len(preview["orphaned_cache"])
            n_orphan_data  = len(preview["orphaned_data"])
            n_orphan_es    = len(preview["orphaned_es"])
            total_orphans  = n_orphan_cache + n_orphan_data + n_orphan_es

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("PDFs sur disque", n_pdfs)
            col_b.metric("Orphelins cache", n_orphan_cache, delta=f"-{n_orphan_cache}" if n_orphan_cache else None, delta_color="inverse")
            col_c.metric("Orphelins data", n_orphan_data, delta=f"-{n_orphan_data}" if n_orphan_data else None, delta_color="inverse")
            col_d.metric("Orphelins ES", n_orphan_es if not preview["es_skipped"] else "N/A")

            if total_orphans == 0:
                st.success("✅ Aucun orphelin — le cache est synchronisé avec les fichiers présents.")
            else:
                st.warning(f"⚠️ **{total_orphans}** entrée(s) orpheline(s) détectée(s) :")
                orphaned_names = list(dict.fromkeys(
                    preview["orphaned_cache"] + preview["orphaned_data"]
                ))
                for fname in orphaned_names:
                    st.markdown(f"  - `{fname}`")

            if preview["es_skipped"]:
                st.caption("ℹ️ Elasticsearch non disponible — la synchronisation ES sera ignorée.")

    except Exception as e:
        st.error(f"❌ Erreur de synchronisation : {e}")

