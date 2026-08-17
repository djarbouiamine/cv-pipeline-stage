#!/usr/bin/env python3
"""
setup_kibana.py
===============
Cree automatiquement le dashboard Kibana complet pour cv-pipeline.
Le dashboard se met a jour en temps reel a chaque CV ajoute.

Usage: python scripts/setup_kibana.py
"""
from __future__ import annotations
import argparse, json, os, sys, time, requests

DEFAULT_KIBANA = os.getenv("KIBANA_URL", "http://localhost:5601")
INDEX_NAME = "cvs"
DATA_VIEW_ID = "cv-pipeline-data-view"
HEADERS = {"kbn-xsrf": "true", "Content-Type": "application/json"}
VIS_IDS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _req(method, kibana_url, path, body=None):
    url = f"{kibana_url.rstrip('/')}{path}"
    try:
        fn = getattr(requests, method)
        kw = {"headers": HEADERS, "timeout": 30}
        if body is not None: kw["json"] = body
        r = fn(url, **kw)
        if r.status_code == 409:
            return r.json()
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"\n[ERREUR] Impossible de joindre Kibana a {kibana_url}.")
        print("         Lancez: docker compose up -d")
        sys.exit(1)
    except requests.HTTPError:
        print(f"\n[ERREUR] HTTP {r.status_code} sur {path}: {r.text[:400]}")
        return {}

def wait_for_kibana(kibana_url, retries=30, delay=5):
    print(f"  Attente de Kibana ...", end="", flush=True)
    for _ in range(retries):
        try:
            r = requests.get(f"{kibana_url}/api/status", timeout=5)
            if r.status_code == 200:
                state = r.json().get("status", {}).get("overall", {}).get("level", "")
                if state in ("available", "degraded"):
                    print(" OK"); return
        except: pass
        print(".", end="", flush=True)
        time.sleep(delay)
    print("\n[ERREUR] Kibana non disponible."); sys.exit(1)

def upsert(kibana_url, obj_type, obj_id, attributes, references=None):
    body = {"attributes": attributes}
    if references: body["references"] = references
    resp = _req("post", kibana_url, f"/api/saved_objects/{obj_type}/{obj_id}?overwrite=true", body)
    return resp.get("id", obj_id)

def search_src():
    return json.dumps({"index": DATA_VIEW_ID, "query": {"language": "kuery", "query": ""}, "filter": []})

def refs():
    return [{"id": DATA_VIEW_ID, "name": "kibanaSavedObjectMeta.searchSourceJSON.index", "type": "index-pattern"}]

def meta(search_source=None):
    return {"kibanaSavedObjectMeta": {"searchSourceJSON": search_source or search_src()}}

# ---------------------------------------------------------------------------
# 1. Data View
# ---------------------------------------------------------------------------
def create_data_view(kibana_url):
    print("  [1/4] Data View 'cvs' ...")
    _req("post", kibana_url, "/api/data_views/data_view", {
        "data_view": {"id": DATA_VIEW_ID, "title": INDEX_NAME, "name": "CV Pipeline — Index cvs"},
        "override": True,
    })

# ---------------------------------------------------------------------------
# 2. Visualizations
# ---------------------------------------------------------------------------
def vis(kibana_url, vis_id, title, vis_type, params, aggs, desc=""):
    attributes = {
        "title": title,
        "visState": json.dumps({"title": title, "type": vis_type, "params": params, "aggs": aggs}),
        "uiStateJSON": "{}",
        "description": desc,
        **meta(),
    }
    upsert(kibana_url, "visualization", vis_id, attributes, refs())
    VIS_IDS[vis_id] = vis_id
    print(f"        {title}")

def create_all_visualizations(kibana_url):
    print("  [2/4] Creation des visualisations ...")

    # ── Metrique : Total CVs ──────────────────────────────────────────────
    vis(kibana_url, "cv-vis-total", "Total CVs indexes", "metric",
        {"type":"metric","addTooltip":True,"addLegend":False,
         "metric":{"percentageMode":False,"useRanges":False,"colorSchema":"Green to Red",
                   "metricColorMode":"None","colorsRange":[{"from":0,"to":10000}],
                   "labels":{"show":True},"invertColors":False,
                   "style":{"bgFill":"#000","bgColor":False,"labelColor":False,"subText":"CVs","fontSize":60}}},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{"customLabel":"Total CVs"}}])

    # ── Metrique : Score moyen global ─────────────────────────────────────
    vis(kibana_url, "cv-vis-score-moyen", "Score qualite moyen", "metric",
        {"type":"metric","addTooltip":True,"addLegend":False,
         "metric":{"percentageMode":False,"useRanges":True,"colorSchema":"Green to Red",
                   "metricColorMode":"Background","colorsRange":[{"from":0,"to":40},{"from":40,"to":70},{"from":70,"to":100}],
                   "labels":{"show":True},"invertColors":False,
                   "style":{"bgFill":"#000","bgColor":True,"labelColor":False,"subText":"/ 100","fontSize":48}}},
        [{"id":"1","enabled":True,"type":"avg","schema":"metric","params":{"field":"score_qualite_globale","customLabel":"Score moyen"}}])

    # ── Camembert : Repartition par categorie ────────────────────────────
    vis(kibana_url, "cv-vis-categories", "Repartition par categorie", "pie",
        {"type":"pie","addTooltip":True,"addLegend":True,"legendPosition":"right","isDonut":True,
         "labels":{"show":True,"values":True,"last_level":True,"truncate":100}},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"categorie_principale","size":10,"order":"desc","orderBy":"1","customLabel":"Categorie"}}])

    # ── Histogramme : Distribution scores ───────────────────────────────
    def bar_params(y_label="Nombre de CVs", x_rotate=0):
        return {
            "type":"histogram","grid":{"categoryLines":False},
            "categoryAxes":[{"id":"CategoryAxis-1","type":"category","position":"bottom","show":True,
                              "style":{},"scale":{"type":"linear"},
                              "labels":{"show":True,"filter":True,"truncate":100,"rotate":x_rotate},"title":{}}],
            "valueAxes":[{"id":"ValueAxis-1","name":"LeftAxis-1","type":"value","position":"left","show":True,
                          "style":{},"scale":{"type":"linear","mode":"normal"},
                          "labels":{"show":True,"rotate":0,"filter":False,"truncate":100},
                          "title":{"text":y_label}}],
            "seriesParams":[{"show":True,"type":"histogram","mode":"stacked",
                             "data":{"label":y_label,"id":"1"},"valueAxis":"ValueAxis-1",
                             "drawLinesBetweenPoints":True,"lineWidth":2,"showCircles":True}],
            "addTooltip":True,"addLegend":True,"legendPosition":"right","times":[],"addTimeMarker":False,
        }

    vis(kibana_url, "cv-vis-scores", "Distribution des scores qualite", "histogram",
        bar_params("Nombre de CVs"),
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"histogram","schema":"segment","params":{"field":"score_qualite_globale","interval":10,"extended_bounds":{},"min_doc_count":False,"customLabel":"Tranche de score"}}])

    # ── Barres horizontales : Top 10 candidats par score ─────────────────
    vis(kibana_url, "cv-vis-top-candidats", "Top 10 candidats par score", "horizontal_bar",
        {"type":"histogram","grid":{"categoryLines":False},
         "categoryAxes":[{"id":"CategoryAxis-1","type":"category","position":"left","show":True,
                          "style":{},"scale":{"type":"linear"},
                          "labels":{"show":True,"filter":True,"truncate":200},"title":{}}],
         "valueAxes":[{"id":"ValueAxis-1","name":"BottomAxis-1","type":"value","position":"bottom","show":True,
                       "style":{},"scale":{"type":"linear","mode":"normal"},
                       "labels":{"show":True,"rotate":0,"filter":False,"truncate":100},
                       "title":{"text":"Score qualite"}}],
         "seriesParams":[{"show":True,"type":"histogram","mode":"stacked",
                          "data":{"label":"Score moyen","id":"1"},"valueAxis":"ValueAxis-1",
                          "drawLinesBetweenPoints":True,"lineWidth":2,"showCircles":True}],
         "addTooltip":True,"addLegend":True,"legendPosition":"right","times":[],"addTimeMarker":False},
        [{"id":"1","enabled":True,"type":"avg","schema":"metric","params":{"field":"score_qualite_globale","customLabel":"Score moyen"}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"nom.keyword","size":10,"order":"desc","orderBy":"1","customLabel":"Candidat"}}])

    # ── Barres : Experience par candidat ─────────────────────────────────
    vis(kibana_url, "cv-vis-experience", "Annees d'experience par candidat", "histogram",
        bar_params("Annees d'experience", x_rotate=30),
        [{"id":"1","enabled":True,"type":"avg","schema":"metric","params":{"field":"annees_experience","customLabel":"Exp. (annees)"}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"nom.keyword","size":20,"order":"desc","orderBy":"1","customLabel":"Candidat"}}])

    # ── Top Technologies (Barres) ─────────────────────────────────────────
    vis(kibana_url, "cv-vis-technologies", "Top Technologies", "horizontal_bar",
        {"type":"histogram","grid":{"categoryLines":False},
         "categoryAxes":[{"id":"CategoryAxis-1","type":"category","position":"left","show":True,
                          "style":{},"scale":{"type":"linear"},
                          "labels":{"show":True,"filter":True,"truncate":200},"title":{}}],
         "valueAxes":[{"id":"ValueAxis-1","name":"BottomAxis-1","type":"value","position":"bottom","show":True,
                       "style":{},"scale":{"type":"linear","mode":"normal"},
                       "labels":{"show":True,"rotate":0,"filter":False,"truncate":100},
                       "title":{"text":"Nombre de candidats"}}],
         "seriesParams":[{"show":True,"type":"histogram","mode":"stacked",
                          "data":{"label":"Candidats","id":"1"},"valueAxis":"ValueAxis-1",
                          "drawLinesBetweenPoints":True,"lineWidth":2,"showCircles":True}],
         "addTooltip":True,"addLegend":True,"legendPosition":"right","times":[],"addTimeMarker":False},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"technologies","size":15,"order":"desc","orderBy":"1","customLabel":"Technologie"}}])

    # ── Top Frameworks ────────────────────────────────────────────────────
    vis(kibana_url, "cv-vis-frameworks", "Top Frameworks", "horizontal_bar",
        {"type":"histogram","grid":{"categoryLines":False},
         "categoryAxes":[{"id":"CategoryAxis-1","type":"category","position":"left","show":True,
                          "style":{},"scale":{"type":"linear"},
                          "labels":{"show":True,"filter":True,"truncate":200},"title":{}}],
         "valueAxes":[{"id":"ValueAxis-1","name":"BottomAxis-1","type":"value","position":"bottom","show":True,
                       "style":{},"scale":{"type":"linear","mode":"normal"},
                       "labels":{"show":True,"rotate":0,"filter":False,"truncate":100},
                       "title":{"text":"Nombre de candidats"}}],
         "seriesParams":[{"show":True,"type":"histogram","mode":"stacked",
                          "data":{"label":"Candidats","id":"1"},"valueAxis":"ValueAxis-1",
                          "drawLinesBetweenPoints":True,"lineWidth":2,"showCircles":True}],
         "addTooltip":True,"addLegend":True,"legendPosition":"right","times":[],"addTimeMarker":False},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"frameworks","size":15,"order":"desc","orderBy":"1","customLabel":"Framework"}}])

    # ── Langues parlees ───────────────────────────────────────────────────
    vis(kibana_url, "cv-vis-langues", "Langues des candidats", "pie",
        {"type":"pie","addTooltip":True,"addLegend":True,"legendPosition":"right","isDonut":False,
         "labels":{"show":True,"values":True,"last_level":True,"truncate":100}},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"langues","size":10,"order":"desc","orderBy":"1","customLabel":"Langue"}}])

    # ── Score par domaine (barres groupees) ───────────────────────────────
    vis(kibana_url, "cv-vis-domaines", "Score moyen par domaine principal", "horizontal_bar",
        {"type":"histogram","grid":{"categoryLines":False},
         "categoryAxes":[{"id":"CategoryAxis-1","type":"category","position":"left","show":True,
                          "style":{},"scale":{"type":"linear"},
                          "labels":{"show":True,"filter":True,"truncate":200},"title":{}}],
         "valueAxes":[{"id":"ValueAxis-1","name":"BottomAxis-1","type":"value","position":"bottom","show":True,
                       "style":{},"scale":{"type":"linear","mode":"normal"},
                       "labels":{"show":True,"rotate":0,"filter":False,"truncate":100},
                       "title":{"text":"Score moyen"}}],
         "seriesParams":[{"show":True,"type":"histogram","mode":"stacked",
                          "data":{"label":"Score moyen","id":"1"},"valueAxis":"ValueAxis-1",
                          "drawLinesBetweenPoints":True,"lineWidth":2,"showCircles":True}],
         "addTooltip":True,"addLegend":True,"legendPosition":"right","times":[],"addTimeMarker":False},
        [{"id":"1","enabled":True,"type":"avg","schema":"metric","params":{"field":"score_qualite_globale","customLabel":"Score moyen"}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"categorie_principale","size":10,"order":"desc","orderBy":"1","customLabel":"Domaine"}}])

    # ── Tableau : Tous les candidats avec description complete ────────────
    search_src_table = json.dumps({
        "index": DATA_VIEW_ID,
        "query": {"language": "kuery", "query": ""},
        "filter": [],
        "sort": [{"score_qualite_globale": {"order": "desc"}}],
    })
    table_attrs = {
        "title": "Tableau complet des candidats",
        "description": "Liste de tous les candidats avec leurs informations completes",
        "columns": ["nom", "categorie_principale", "score_qualite_globale", "annees_experience",
                    "langages", "frameworks", "technologies", "langues", "localisation"],
        "sort": [["score_qualite_globale", "desc"]],
        "kibanaSavedObjectMeta": {"searchSourceJSON": search_src_table},
    }
    upsert(kibana_url, "search", "cv-search-table", table_attrs, refs())
    VIS_IDS["cv-search-table"] = "cv-search-table"
    print("        Tableau : Tous les candidats (decouverte)")

    # ── Tagcloud : Technologies les plus frequentes ───────────────────────
    vis(kibana_url, "cv-vis-tagcloud", "Nuage de competences", "tagcloud",
        {"scale":"linear","orientation":"single","minFontSize":18,"maxFontSize":72,
         "showLabel":False},
        [{"id":"1","enabled":True,"type":"count","schema":"metric","params":{}},
         {"id":"2","enabled":True,"type":"terms","schema":"segment","params":{"field":"technologies","size":30,"order":"desc","orderBy":"1","customLabel":"Competence"}}])

    # ── Gauge : Score qualite moyen ───────────────────────────────────────
    vis(kibana_url, "cv-vis-gauge", "Jauge Score Qualite", "gauge",
        {"type":"gauge","addTooltip":True,"addLegend":False,
         "isDisplayWarning":False,
         "gauge":{"verticalSplit":False,"extendRange":True,"percentageMode":True,
                  "gaugeType":"Arc","gaugeStyle":"Full","backStyle":"Full",
                  "orientation":"vertical","colorSchema":"Green to Red",
                  "gaugeColorMode":"Labels","colorsRange":[{"from":0,"to":0.4},{"from":0.4,"to":0.7},{"from":0.7,"to":1}],
                  "invertColors":False,"labels":{"show":True,"color":"black"},
                  "scale":{"show":True,"labels":False,"color":"rgba(105,112,125,0.2)"},
                  "type":"meter","style":{"bgFill":"rgba(105,112,125,0.2)","bgColor":False,"labelColor":False,"subText":"","fontSize":60},
                  "minAngle":0.3926990816987242,"maxAngle":2.748893571891069,
                  "alignment":"automatic"}},
        [{"id":"1","enabled":True,"type":"avg","schema":"metric","params":{"field":"score_qualite_globale","customLabel":"Score moyen / 100"}}])


# ---------------------------------------------------------------------------
# 3. Dashboard final
# ---------------------------------------------------------------------------
DASHBOARD_ID = "cv-pipeline-dashboard"

def create_dashboard(kibana_url):
    print("  [3/4] Assemblage du Dashboard ...")
    panels = [
        # Ligne 1 : KPIs
        {"panelIndex":"p1","gridData":{"x":0,  "y":0,  "w":8,  "h":7, "i":"p1"},"type":"visualization","panelRefName":"ref_total"},
        {"panelIndex":"p2","gridData":{"x":8,  "y":0,  "w":8,  "h":7, "i":"p2"},"type":"visualization","panelRefName":"ref_score_moyen"},
        {"panelIndex":"p3","gridData":{"x":16, "y":0,  "w":8,  "h":7, "i":"p3"},"type":"visualization","panelRefName":"ref_gauge"},
        {"panelIndex":"p4","gridData":{"x":24, "y":0,  "w":24, "h":16,"i":"p4"},"type":"visualization","panelRefName":"ref_categories"},
        # Ligne 2 : Scores + Experience
        {"panelIndex":"p5","gridData":{"x":0,  "y":7,  "w":24, "h":14,"i":"p5"},"type":"visualization","panelRefName":"ref_scores"},
        # Ligne 3 : Top candidats
        {"panelIndex":"p6","gridData":{"x":0,  "y":21, "w":48, "h":16,"i":"p6"},"type":"visualization","panelRefName":"ref_top"},
        # Ligne 4 : Tech + Frameworks
        {"panelIndex":"p7","gridData":{"x":0,  "y":37, "w":24, "h":18,"i":"p7"},"type":"visualization","panelRefName":"ref_tech"},
        {"panelIndex":"p8","gridData":{"x":24, "y":37, "w":24, "h":18,"i":"p8"},"type":"visualization","panelRefName":"ref_frameworks"},
        # Ligne 5 : Domaines + Langues + Nuage
        {"panelIndex":"p9", "gridData":{"x":0,  "y":55, "w":16, "h":16,"i":"p9"}, "type":"visualization","panelRefName":"ref_domaines"},
        {"panelIndex":"p10","gridData":{"x":16, "y":55, "w":16, "h":16,"i":"p10"},"type":"visualization","panelRefName":"ref_langues"},
        {"panelIndex":"p11","gridData":{"x":32, "y":55, "w":16, "h":16,"i":"p11"},"type":"visualization","panelRefName":"ref_tagcloud"},
        # Ligne 6 : Experience + Tableau
        {"panelIndex":"p12","gridData":{"x":0,  "y":71, "w":24, "h":16,"i":"p12"},"type":"visualization","panelRefName":"ref_experience"},
        {"panelIndex":"p13","gridData":{"x":0,  "y":87, "w":48, "h":20,"i":"p13"},"type":"search","panelRefName":"ref_table"},
    ]
    references = [
        {"id":"cv-vis-total",       "name":"ref_total",       "type":"visualization"},
        {"id":"cv-vis-score-moyen", "name":"ref_score_moyen", "type":"visualization"},
        {"id":"cv-vis-gauge",       "name":"ref_gauge",       "type":"visualization"},
        {"id":"cv-vis-categories",  "name":"ref_categories",  "type":"visualization"},
        {"id":"cv-vis-scores",      "name":"ref_scores",      "type":"visualization"},
        {"id":"cv-vis-top-candidats","name":"ref_top",        "type":"visualization"},
        {"id":"cv-vis-technologies","name":"ref_tech",        "type":"visualization"},
        {"id":"cv-vis-frameworks",  "name":"ref_frameworks",  "type":"visualization"},
        {"id":"cv-vis-domaines",    "name":"ref_domaines",    "type":"visualization"},
        {"id":"cv-vis-langues",     "name":"ref_langues",     "type":"visualization"},
        {"id":"cv-vis-tagcloud",    "name":"ref_tagcloud",    "type":"visualization"},
        {"id":"cv-vis-experience",  "name":"ref_experience",  "type":"visualization"},
        {"id":"cv-search-table",    "name":"ref_table",       "type":"search"},
    ]
    attrs = {
        "title": "CV Pipeline Dashboard",
        "description": "Dashboard enterprise — KPIs, scores, competences, tableau candidats. Se met a jour automatiquement a chaque ajout de CV.",
        "panelsJSON": json.dumps(panels),
        "timeRestore": False,
        "refreshInterval": {"pause": False, "value": 30000},
        "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps({"query":{"language":"kuery","query":""},"filter":[]})},
    }
    upsert(kibana_url, "dashboard", DASHBOARD_ID, attrs, references)
    print(f"     URL : {kibana_url}/app/dashboards#/view/{DASHBOARD_ID}")

# ---------------------------------------------------------------------------
# 4. Exporter le NDJSON (pour le repo Git)
# ---------------------------------------------------------------------------
def export_ndjson(kibana_url):
    print("  [4/4] Export NDJSON (sauvegarde dans le repo) ...")
    types = ["dashboard","visualization","search","index-pattern"]
    out = os.path.join(os.path.dirname(__file__), "..", "kibana_dashboard.ndjson")
    try:
        r = requests.post(
            f"{kibana_url}/api/saved_objects/_export",
            headers=HEADERS,
            json={"type": types, "includeReferencesDeep": True},
            timeout=30,
        )
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
        n = r.content.count(b"\n")
        print(f"     {n} objets exportes -> kibana_dashboard.ndjson")
    except Exception as e:
        print(f"     [AVERTISSEMENT] Export NDJSON echoue : {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kibana", default=DEFAULT_KIBANA)
    args = parser.parse_args()
    print("\n=== Setup Dashboard Kibana — CV Pipeline ===\n")
    wait_for_kibana(args.kibana)
    create_data_view(args.kibana)
    create_all_visualizations(args.kibana)
    create_dashboard(args.kibana)
    export_ndjson(args.kibana)
    print("\n=== Dashboard pret ! ===")
    print(f"  {args.kibana}/app/dashboards")
    print("  Le dashboard se met a jour automatiquement (refresh 30s)")

if __name__ == "__main__":
    main()
