"""Classement final par adéquation poste (pas seulement score qualité global).

CORRECTIF (généralisation à n'importe quelle question) :
Avant, le poids "compétences" (40%, le plus gros du score) ne reconnaissait
que ~35 technologies codées en dur dans KNOWN_SKILLS. Toute question qui
n'utilisait aucun de ces mots exacts (ex: "chef de projet", "data analyst",
"réseaux télécoms") faisait retomber le matching sur une valeur neutre pour
TOUS les candidats -> le classement dépendait alors surtout de la qualité
générale du CV, pas de sa pertinence réelle pour la question posée.

Deux changements :
1. Le poids sémantique (la vraie similarité vectorielle, qui comprend
   n'importe quel vocabulaire) est remonté au-dessus du poids "compétences".
2. Le matching de compétences est complété par un matching générique par
   mots-clés qui compare directement les mots de la question au vocabulaire
   structuré de chaque CV (technologies, langages, frameworks, certifs,
   domaine) -- ça fonctionne pour n'importe quel terme, pas seulement ceux
   de la liste fixe.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Final Score = 30% skills (liste connue + vocabulaire générique du CV)
#             + 35% semantic (similarité vectorielle réelle)
#             + 15% quality + 10% exp + 5% cert + 5% projects
W_SKILL = 0.30
W_SEMANTIC = 0.35
W_QUALITY = 0.15
W_EXPERIENCE = 0.10
W_CERT = 0.05
W_PROJECTS = 0.05

# Mots trop courants pour être un signal de compétence/métier -> ignorés
# par le matching générique (sinon "le", "avec", "pour"... compteraient
# comme un "mot-clé" de la question et fausseraient le ratio).
_STOPWORDS_FR_EN = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au",
    "aux", "en", "dans", "sur", "pour", "avec", "sans", "qui", "que", "quoi",
    "est", "sont", "être", "avoir", "ce", "cet", "cette", "ces", "je", "tu",
    "il", "elle", "on", "nous", "vous", "ils", "elles", "mon", "ton", "son",
    "trouve", "trouver", "cherche", "recherche", "recherché", "moi", "candidat",
    "candidate", "candidats", "profil", "profils", "cv", "the", "and", "or",
    "for", "with", "without", "who", "what", "is", "are", "be", "this",
    "that", "these", "those", "find", "search", "looking", "profile", "me",
    "meilleur", "meilleure", "bon", "bonne", "compétent",
    "compétente", "experience", "expérience", "ans", "année", "années",
}


def _content_tokens(text: str) -> List[str]:
    """Découpe un texte en mots significatifs (>=3 lettres, hors stopwords)."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Zà-ÿÀ-Ÿ0-9+#\.]{2,}", text.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS_FR_EN]


def _doc_vocab_tokens(doc: Dict) -> set:
    """Rassemble tous les mots issus des champs structurés du CV
    (technologies, langages, frameworks, outils, certifs, domaine) --
    c'est le vocabulaire "officiel" extrait par le LLM pour ce candidat,
    donc plus fiable que de chercher dans le texte brut.
    """
    vocab: set = set()
    for field in ("technologies", "langages", "frameworks", "outils_devops",
                  "certifications", "categorie_principale"):
        v = doc.get(field)
        if isinstance(v, list):
            for item in v:
                vocab.update(_content_tokens(str(item)))
        elif v:
            vocab.update(_content_tokens(str(v)))
    return vocab


def dynamic_match_ratio(question: str, doc: Dict) -> float:
    """Proportion des mots-clés significatifs de la question retrouvés
    dans le vocabulaire structuré du CV. Fonctionne pour N'IMPORTE QUEL
    terme (pas seulement ceux de KNOWN_SKILLS), car il compare directement
    les mots de la question au contenu réel de chaque CV.
    """
    q_tokens = set(_content_tokens(question))
    if not q_tokens:
        return 1.0
    vocab = _doc_vocab_tokens(doc)
    if not vocab:
        return 0.0
    hits = 0
    for tok in q_tokens:
        if tok in vocab or any(tok in v or v in tok for v in vocab):
            hits += 1
    return hits / len(q_tokens)

KNOWN_SKILLS: List[Tuple[str, str, str]] = [
    # (pattern, label, category)
    (r"\bdocker\b", "Docker", "devops"),
    (r"\bkubernetes\b|\bk8s\b", "Kubernetes", "devops"),
    (r"\bterraform\b", "Terraform", "devops"),
    (r"\baws\b|amazon web services", "AWS", "cloud"),
    (r"\bazure\b", "Azure", "cloud"),
    (r"\bgcp\b|google cloud", "GCP", "cloud"),
    (r"\bci/?cd\b", "CI/CD", "devops"),
    (r"\bpython\b", "Python", "lang"),
    (r"\bjava\b", "Java", "lang"),
    (r"\bjavascript\b|\bjs\b", "JavaScript", "lang"),
    (r"\btypescript\b", "TypeScript", "lang"),
    (r"\bc\+\+\b", "C++", "lang"),
    (r"\bc#\b", "C#", "lang"),
    # Boundary assouplie (pas de \b final) : "React" doit aussi matcher
    # "ReactJS" / "React.js" écrits collés, sinon ces CV étaient à tort
    # comptés comme "ne maîtrise pas React".
    (r"\breact(js|\.js|native)?\b", "React", "frontend"),
    (r"\bangular(js)?\b", "Angular", "frontend"),
    (r"\bvue(js|\.js)?\b", "Vue", "frontend"),
    (r"\bflutter\b", "Flutter", "mobile"),
    (r"\btensorflow\b", "TensorFlow", "ai"),
    (r"\bpytorch\b", "PyTorch", "ai"),
    (r"\bnlp\b|natural language", "NLP", "ai"),
    (r"computer vision|vision par ordinateur", "Computer Vision", "ai"),
    (r"machine learning|deep learning|\bml\b|\bia\b|\bai\b", "Machine Learning / IA", "ai"),
    (r"\bspring boot\b|\bspring\b", "Spring Boot", "backend"),
    (r"\bgolang\b|\bgo\b", "Go", "lang"),
    (r"backend|back-end", "Backend", "backend"),
    (r"frontend|front-end", "Frontend", "frontend"),
    (r"\bsql\b", "SQL", "data"),
    (r"\bmongodb\b", "MongoDB", "data"),
    (r"\bpostgresql\b|\bpostgres\b", "PostgreSQL", "data"),
    (r"\brust\b", "Rust", "lang"),
    (r"blockchain|solidity|web3", "Blockchain", "other"),
    (r"cybersécurité|cybersecurite|cyber security", "Cybersécurité", "security"),
    (r"\blinux\b", "Linux", "devops"),
    (r"\bnode\.?js\b", "Node.js", "backend"),
    (r"\bfastapi\b", "FastAPI", "backend"),
    (r"\bflask\b", "Flask", "backend"),
    (r"\bstreamlit\b", "Streamlit", "other"),
    (r"embedded|embarqu", "Embedded", "embedded"),
    (r"quantum\s*computing|quantum", "Quantum Computing", "other"),
    (r"\btensorflow\b", "TensorFlow", "ai"),
    (r"\bkeras\b", "Keras", "ai"),
    (r"\bscikit\b|\bsklearn\b", "Scikit-learn", "ai"),
    (r"\bpandas\b", "Pandas", "data"),
    (r"\bnumpy\b", "NumPy", "data"),
    (r"\bmatplotlib\b", "Matplotlib", "data"),
    (r"\bpowerbi\b|power bi", "Power BI", "data"),
    (r"\btableau\b", "Tableau", "data"),
    (r"\bjira\b", "Jira", "devops"),
    (r"\bgit\b", "Git", "devops"),
    (r"\bjenkins\b", "Jenkins", "devops"),
    (r"\bansible\b", "Ansible", "devops"),
    (r"\bgrafana\b", "Grafana", "devops"),
    (r"\breact native\b", "React Native", "mobile"),
    (r"\bswift\b", "Swift", "mobile"),
    (r"\bkotlin\b", "Kotlin", "mobile"),
    (r"\bdart\b", "Dart", "mobile"),
    (r"\bscala\b", "Scala", "lang"),
    (r"\bruby\b", "Ruby", "lang"),
    (r"\bphp\b", "PHP", "lang"),
    (r"\bc\b", "C", "lang"),
    (r"\bgitlab\b", "GitLab", "devops"),
    (r"\bnginx\b", "Nginx", "devops"),
    (r"\bredis\b", "Redis", "data"),
    (r"\belasticsearch\b", "Elasticsearch", "data"),
    (r"\bkafka\b", "Kafka", "data"),
    (r"\bspark\b|apache spark", "Apache Spark", "data"),
    (r"\bhadoop\b", "Hadoop", "data"),
    (r"\bsql server\b|mssql", "SQL Server", "data"),
    (r"\bmysql\b", "MySQL", "data"),
    (r"\bcassandra\b", "Cassandra", "data"),
    (r"\bdynamodb\b", "DynamoDB", "data"),
    (r"\bselenium\b", "Selenium", "other"),
    (r"\bpostman\b", "Postman", "other"),
    (r"\bfigma\b", "Figma", "other"),
    (r"\bsass\b|\bscss\b", "SASS/SCSS", "frontend"),
    (r"\bnext\.?js\b", "Next.js", "frontend"),
    (r"\bnuxt\b", "Nuxt", "frontend"),
    (r"\btailwind\b", "Tailwind CSS", "frontend"),
    (r"\bbootstrap\b", "Bootstrap", "frontend"),
    (r"\blaravel\b", "Laravel", "backend"),
    (r"\bdjango\b", "Django", "backend"),
    (r"\bspring\b|spring boot", "Spring Boot", "backend"),
    (r"\bexpress\b|express\.js", "Express.js", "backend"),
    (r"\b\.net\b|dotnet", ".NET", "backend"),
]


def extract_query_skills(question: str) -> List[str]:
    """Compétences explicitement mentionnées dans la question."""
    q = question.lower()
    found: List[str] = []
    seen: set = set()
    for pattern, label, _cat in KNOWN_SKILLS:
        if re.search(pattern, q, re.IGNORECASE):
            if label not in seen:
                seen.add(label)
                found.append(label)
    return found


def question_requires_all_skills(question: str) -> bool:
    """True si la question exige une adéquation stricte (ingénieur, avec, AND, etc.)."""
    q = question.lower()
    if " and " in q or " et " in q:
        return True
    if re.search(r"\b(ingénieur|ingenieur|engineer|developer|développeur|developpeur)\b", q):
        return True
    if re.search(r"\b(must|obligatoire|requiert|require|mandatory|satisfy)\b", q):
        return True
    return False


def _doc_haystack(doc: Dict) -> str:
    parts: List[str] = []
    for key in ("technologies", "langages", "frameworks", "outils_devops", "certifications", "text"):
        v = doc.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    proj = doc.get("projets") or []
    if isinstance(proj, list):
        parts.extend(str(p) for p in proj)
    return " ".join(parts).lower()


def _skill_present(haystack: str, label: str) -> bool:
    for pattern, lbl, _ in KNOWN_SKILLS:
        if lbl == label and re.search(pattern, haystack, re.IGNORECASE):
            return True
    return False


def skill_match_ratio(doc: Dict, skills: List[str]) -> float:
    if not skills:
        return 1.0
    h = _doc_haystack(doc)
    hits = sum(1 for s in skills if _skill_present(h, s))
    return hits / len(skills)


def missing_skills(doc: Dict, skills: List[str]) -> List[str]:
    if not skills:
        return []
    h = _doc_haystack(doc)
    return [s for s in skills if not _skill_present(h, s)]


def _normalize_semantic(raw: Optional[float], max_raw: float) -> float:
    if raw is None or max_raw <= 0:
        return 0.0
    return min(1.0, max(0.0, float(raw) / max_raw))


def _experience_norm(doc: Dict) -> float:
    exp = doc.get("annees_experience")
    if not isinstance(exp, (int, float)):
        return 0.0
    return min(1.0, float(exp) / 10.0)


def _cert_norm(doc: Dict) -> float:
    certs = doc.get("certifications") or []
    if isinstance(certs, str):
        certs = [certs] if certs.strip() else []
    n = len(certs) if isinstance(certs, list) else 0
    return min(1.0, n / 4.0)


def _project_norm(doc: Dict) -> float:
    proj = doc.get("projets") or []
    if isinstance(proj, str):
        proj = [proj] if proj.strip() else []
    n = len(proj) if isinstance(proj, list) else 0
    return min(1.0, n / 5.0)


def _quality_norm(doc: Dict) -> float:
    q = doc.get("score_qualite_globale")
    if not isinstance(q, (int, float)):
        return 0.0
    return min(1.0, max(0.0, float(q) / 100.0))


def compute_final_score(
    question: str,
    doc: Dict,
    max_semantic: float,
    query_skills: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], List[str], List[str]]:
    """
    Retourne (score 0-100, composantes 0-100, forces, compétences manquantes).
    """
    skills = query_skills if query_skills is not None else extract_query_skills(question)
    dyn = dynamic_match_ratio(question, doc)
    if skills:
        # Compétence(s) reconnue(s) dans la liste connue : on combine le
        # matching précis (regex ciblée) avec le matching générique, pour
        # ne pas perdre le signal si la question mentionne AUSSI des
        # termes hors liste (ex: "Python avec expérience en fintech").
        sm = 0.6 * skill_match_ratio(doc, skills) + 0.4 * dyn
    else:
        # Rien dans la liste fixe -> on s'appuie entièrement sur le
        # matching générique, qui fonctionne avec n'importe quel
        # vocabulaire métier/technique présent dans les CV.
        sm = dyn
    sem = _normalize_semantic(doc.get("_retrieval_score"), max_semantic)
    qual = _quality_norm(doc)
    exp = _experience_norm(doc)
    cert = _cert_norm(doc)
    proj = _project_norm(doc)

    final = (
        W_SKILL * sm
        + W_SEMANTIC * sem
        + W_QUALITY * qual
        + W_EXPERIENCE * exp
        + W_CERT * cert
        + W_PROJECTS * proj
    ) * 100.0

    components = {
        "skill_match": round(sm * 100, 1),
        "semantic": round(sem * 100, 1),
        "quality": round(qual * 100, 1),
        "experience": round(exp * 100, 1),
        "certifications": round(cert * 100, 1),
        "projects": round(proj * 100, 1),
    }

    strengths: List[str] = []
    miss = missing_skills(doc, skills) if skills else []
    for s in skills:
        if s not in miss:
            strengths.append(s)
    if qual >= 0.7:
        strengths.append("bon score qualité CV")
    n_proj = len(doc.get("projets") or []) if isinstance(doc.get("projets"), list) else 0
    if n_proj >= 2:
        strengths.append(f"{n_proj} projets")

    return round(final, 1), components, strengths[:8], miss


def rerank_docs_by_job_relevance(question: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return docs
    query_skills = extract_query_skills(question)
    max_sem = max((d.get("_retrieval_score") or 0) for d in docs) or 1.0
    scored: List[Dict] = []
    for d in docs:
        d = dict(d)
        final, comp, plus, miss = compute_final_score(question, d, max_sem, query_skills)
        d["_match_score"] = final
        d["_match_components"] = comp
        d["_match_strengths"] = plus
        d["_missing_skills"] = miss
        d["_selection_reasons"] = _selection_reasons(plus, comp, query_skills, miss)
        scored.append(d)
    scored.sort(
        key=lambda x: (
            -x["_match_score"],
            -(x.get("_retrieval_score") or 0),
            -(x.get("score_qualite_globale") or 0),
        )
    )
    return scored


def _selection_reasons(
    strengths: List[str],
    comp: Dict[str, float],
    query_skills: List[str],
    missing: List[str],
) -> List[str]:
    reasons: List[str] = []
    if query_skills and not missing:
        reasons.append("correspondance compétences demandées")
    elif query_skills and strengths:
        reasons.append(f"compétences partielles ({', '.join(strengths[:3])})")
    if comp.get("semantic", 0) >= 60:
        reasons.append("forte similarité sémantique")
    if comp.get("quality", 0) >= 70:
        reasons.append("score qualité élevé")
    if comp.get("projects", 0) >= 50:
        reasons.append("projets pertinents")
    if comp.get("certifications", 0) >= 50:
        reasons.append("certifications")
    return reasons[:6]


def split_exact_skill_matches(
    question: str,
    docs: List[Dict],
) -> Tuple[List[Dict], List[Dict], bool]:
    """
    Sépare correspondances exactes (toutes compétences) vs profils proches.
    """
    skills = extract_query_skills(question)
    if not skills or not question_requires_all_skills(question):
        return docs, [], False

    exact: List[Dict] = []
    others: List[Dict] = []
    for d in docs:
        miss = missing_skills(d, skills)
        if not miss:
            exact.append(d)
        else:
            d = dict(d)
            d["_missing_skills"] = miss
            d["_reject_reason"] = "Compétences manquantes : " + ", ".join(miss)
            others.append(d)
    return exact, others, True