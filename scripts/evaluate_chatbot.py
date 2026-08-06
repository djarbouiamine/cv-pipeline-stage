#!/usr/bin/env python3
"""Evaluate chatbot routing & retrieval against the test question suite."""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Mock streamlit before importing project modules
def _noop(*a, **k):
    return None


class _Sidebar:
    selectbox = staticmethod(lambda *a, **k: "groq")
    radio = staticmethod(lambda *a, **k: "auto")
    divider = staticmethod(_noop)
    button = staticmethod(lambda *a, **k: False)
    markdown = staticmethod(_noop)


class _ChatMessage:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def markdown(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def json(self, *a, **k):
        pass

    def dataframe(self, *a, **k):
        pass

    def expander(self, *a, **k):
        return self


st_mock = types.ModuleType("streamlit")
st_mock.cache_resource = lambda **kw: (lambda fn: fn)
st_mock.warning = _noop
st_mock.set_page_config = _noop
st_mock.title = _noop
st_mock.caption = _noop
st_mock.sidebar = _Sidebar()
st_mock.divider = _noop
st_mock.markdown = _noop
st_mock.chat_message = lambda *a, **k: _ChatMessage()
st_mock.chat_input = lambda *a, **k: None
st_mock.spinner = lambda *a, **k: type("S", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


st_mock.session_state = _SessionState()
st_mock.rerun = _noop
sys.modules["streamlit"] = st_mock

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from elasticsearch import Elasticsearch  # noqa: E402

from chatbot.intent import classify_intent_heuristic  # noqa: E402
from chatbot.category_resolver import detect_topic_in_question, get_known_categories  # noqa: E402
from chatbot.mandatory_criteria import detect_mandatory_requirements  # noqa: E402
from chatbot.es_aggregations import detect_aggregation_intent  # noqa: E402
from chatbot.final_ranking import extract_query_skills  # noqa: E402

# Import chatbot page helpers (route_question, etc.)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "chatbot_page",
    os.path.join(ROOT, "pages", "3_Chatbot.py"),
)
chatbot_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chatbot_page)


def _fast_classify(question, known_categories, known_names):
    """Deterministic classifier for batch eval (no LLM)."""
    q = question.lower()
    if chatbot_page.is_comparison_question(question, None):
        return {
            "mode": "comparison", "sort_field": None, "category": None,
            "candidate_names": [], "top_n": None, "confidence": 0.9,
        }
    if chatbot_page.contains_keyword(q, chatbot_page.MOTS_CLES_STATS):
        return {
            "mode": "stats", "sort_field": None, "category": None,
            "candidate_names": [], "top_n": None, "confidence": 0.9,
        }
    if chatbot_page.contains_keyword(q, chatbot_page.MOTS_CLES_CLASSEMENT):
        return {
            "mode": "classement", "sort_field": "score_qualite_globale",
            "category": chatbot_page.detect_category_in_question(question),
            "candidate_names": [], "top_n": chatbot_page.extract_top_n(q),
            "confidence": 0.9,
        }
    if chatbot_page.contains_keyword(q, chatbot_page.MOTS_CLES_RANKING_ASC):
        sf, _ = chatbot_page.detect_sort_field(question)
        return {
            "mode": "ranking_asc", "sort_field": sf, "category": None,
            "candidate_names": [], "top_n": None, "confidence": 0.9,
        }
    if chatbot_page.contains_keyword(q, chatbot_page.MOTS_CLES_RANKING):
        sf, _ = chatbot_page.detect_sort_field(question)
        return {
            "mode": "ranking_desc", "sort_field": sf, "category": None,
            "candidate_names": [], "top_n": None, "confidence": 0.9,
        }
    if chatbot_page.contains_keyword(q, chatbot_page.MOTS_CLES_FILTRAGE):
        return {
            "mode": "filter", "sort_field": None,
            "category": chatbot_page.detect_category_in_question(question),
            "candidate_names": [], "top_n": None, "confidence": 0.9,
        }
    return {
        "mode": "semantic", "sort_field": None,
        "category": chatbot_page.detect_category_in_question(question),
        "candidate_names": [], "top_n": None, "confidence": 0.8,
    }


chatbot_page.classify_question = _fast_classify


def get_es() -> Elasticsearch:
    host = os.getenv("ELASTIC_HOST", "http://localhost:9200")
    es = Elasticsearch(hosts=[host], timeout=15)
    if not es.ping():
        raise ConnectionError(f"Elasticsearch not available at {host}")
    return es


TEST_QUESTIONS: Dict[str, List[str]] = {
    "semantic_search": [
        "Find the best AI candidate.",
        "Find an Embedded Systems engineer.",
        "Find a Computer Vision engineer.",
        "Find an NLP specialist.",
        "Find a Machine Learning engineer.",
        "Find an IoT engineer.",
        "Find a Data Engineer.",
        "Find a Full-Stack developer.",
        "Find a Backend developer.",
        "Find a Frontend developer.",
    ],
    "skills_search": [
        "Who knows Python?",
        "Who knows Java?",
        "Who knows C++?",
        "Who knows SQL?",
        "Who knows TensorFlow?",
        "Who knows PyTorch?",
        "Who knows Docker?",
        "Who knows Kubernetes?",
        "Who knows Terraform?",
        "Who knows React?",
        "Who knows Angular?",
        "Who knows Spring Boot?",
        "Who knows Flutter?",
        "Who knows MongoDB?",
        "Who knows PostgreSQL?",
    ],
    "project_search": [
        "Find candidates who built AI projects.",
        "Find candidates with Computer Vision projects.",
        "Find candidates with NLP projects.",
        "Find candidates with web development projects.",
        "Find candidates with mobile app projects.",
        "Find candidates with cybersecurity projects.",
        "Find candidates with embedded projects.",
        "Find candidates with cloud projects.",
    ],
    "recruiter": [
        "Who should I hire for an AI internship?",
        "Recommend a Backend developer.",
        "Recommend a DevOps engineer.",
        "Recommend a Cybersecurity analyst.",
        "Recommend a Computer Vision engineer.",
        "Recommend an IoT engineer.",
        "Recommend someone for a startup.",
        "Recommend the strongest overall candidate.",
    ],
    "comparison": [
        "Compare Ahmed Abdelhedi and Louati Oussema.",
        "Compare the top 3 AI candidates.",
        "Who is stronger in Python?",
        "Who has more AI projects?",
        "Who has more certifications?",
        "Who has more experience?",
        "Which candidate has the broadest skill set?",
        "Which profile is the most complete?",
    ],
    "statistics": [
        "How many AI candidates are there?",
        "How many Embedded candidates are there?",
        "How many candidates know Python?",
        "How many candidates know Docker?",
        "Which technology appears most often?",
        "Which programming language is most common?",
        "Which framework is the most common?",
        "What is the average quality score?",
        "What is the average years of experience?",
        "Show the distribution of categories.",
    ],
    "ranking": [
        "Show the top 5 candidates.",
        "Rank AI candidates by score.",
        "Rank candidates by experience.",
        "Rank candidates by number of projects.",
        "Rank candidates by certifications.",
        "Show the highest-scoring candidate.",
    ],
    "education": [
        "Who has a CCNA certification?",
        "Who has Red Hat certifications?",
        "Who has machine learning certifications?",
        "Which candidates studied at ENSI?",
        "Which candidates studied at SUP'COM?",
        "Which candidates have engineering degrees?",
    ],
    "edge_cases": [
        "Find an AWS-certified engineer.",
        "Find an Azure expert.",
        "Find a GCP expert.",
        "Find a blockchain developer.",
        "Find a Rust developer.",
        "Find a Go developer.",
        "Find someone with 5 years of experience.",
        "Find someone who speaks Japanese.",
        "Find someone with a PhD.",
        "Find a senior DevOps engineer.",
    ],
    "multi_step": [
        "Who is the best AI candidate with Python and PyTorch?",
        "Find an Embedded engineer who also knows AI.",
        "Compare the best AI candidate with the best Cybersecurity candidate.",
        "Recommend the best Computer Vision intern with TensorFlow.",
        "Find candidates with Python but without Java.",
        "Which AI candidate has the highest score and the most projects?",
        "Show candidates with Docker and Kubernetes ranked by experience.",
        "Which candidate is best suited for an IoT startup?",
    ],
    "general_chat": [
        "What is Python?",
        "Explain machine learning.",
        "What is Docker?",
        "What is Kubernetes?",
        "Explain RAG.",
        "What is Elasticsearch?",
        "Hello!",
        "How are you?",
        "Can you explain deep learning?",
    ],
}


@dataclass
class EvalResult:
    category: str
    question: str
    intent: str
    route_mode: str
    route_desc: str
    n_docs: int
    doc_names: List[str]
    has_stats: bool
    topic: Optional[str]
    mandatory: List[str]
    agg_intent: Optional[str]
    no_exact_match: bool = False
    issues: List[str] = field(default_factory=list)


def evaluate_question(
    es,
    category: str,
    question: str,
    known_names: List[str],
) -> EvalResult:
    issues: List[str] = []

    intent = classify_intent_heuristic(question, known_names)
    if intent is None:
        intent = "ambiguous→general_default"

    topic = detect_topic_in_question(question)
    mandatory = [r.user_facing for r in detect_mandatory_requirements(question)]
    agg_intent = detect_aggregation_intent(question)
    skills = extract_query_skills(question)

    route_mode = "general"
    route_desc = "N/A (general intent)"
    n_docs = 0
    doc_names: List[str] = []
    has_stats = False
    no_exact_match = False

    if intent == "cv" or intent == "ambiguous→general_default":
        # For ambiguous, chatbot defaults to general — flag if likely CV question
        if intent == "ambiguous→general_default":
            cv_signals = bool(topic or skills or mandatory or agg_intent)
            if cv_signals:
                issues.append("INTENT: likely CV question but heuristic returned ambiguous (defaults to general)")

        if intent == "cv":
            try:
                docs, stats, route_mode, route_desc = chatbot_page.route_question(question, es)
                has_stats = stats is not None
                n_docs = len(docs) if docs else 0
                doc_names = [d.get("nom", "?") for d in (docs or [])[:5]]

                if route_mode == "semantic":
                    docs, meta = chatbot_page.enrich_semantic_retrieval(question, docs or [])
                    no_exact_match = meta.get("no_exact_match", False)
                    n_docs = len(docs)
                    doc_names = [d.get("nom", "?") for d in docs[:5]]

                if route_mode != "stats" and n_docs == 0:
                    issues.append("RETRIEVAL: zero documents returned")

                if mandatory and route_mode == "semantic" and not no_exact_match:
                    issues.append("EDGE: mandatory criteria detected but no_exact_match not set")

                if mandatory and no_exact_match and n_docs > 0:
                    pass  # expected behavior

            except Exception as e:
                issues.append(f"ROUTE_ERROR: {e}")

    # General chat should NOT route to CV
    if category == "general_chat" and intent == "cv":
        issues.append("INTENT: general question misclassified as CV")

    # Edge cases with mandatory criteria
    if category == "edge_cases" and mandatory:
        if intent != "cv":
            issues.append("INTENT: edge case should be CV mode")
        if not no_exact_match and n_docs > 0 and route_mode == "semantic":
            # May still be OK if someone actually has the skill
            pass

    # Comparison with names
    if category == "comparison" and "compare" in question.lower():
        if "ahmed" in question.lower() or "louati" in question.lower():
            if route_mode != "comparison":
                issues.append(f"COMPARISON: expected comparison mode, got {route_mode}")
            elif n_docs < 2:
                issues.append("COMPARISON: fewer than 2 candidates found for named comparison")

    # Stats questions
    if category == "statistics":
        if intent != "cv":
            issues.append("INTENT: stats question should be CV")
        if route_mode not in ("stats", "filter", "semantic", "classement", "ranking_desc"):
            issues.append(f"STATS: unexpected mode {route_mode}")

    return EvalResult(
        category=category,
        question=question,
        intent=intent or "general",
        route_mode=route_mode,
        route_desc=route_desc[:80],
        n_docs=n_docs,
        doc_names=doc_names,
        has_stats=has_stats,
        topic=topic,
        mandatory=mandatory,
        agg_intent=agg_intent,
        no_exact_match=no_exact_match,
        issues=issues,
    )


def main():
    es = get_es()
    total = es.count(index="cvs")["count"]
    categories = get_known_categories(es)
    known_names = chatbot_page.get_all_candidate_names(es)

    print(f"=== Chatbot Evaluation ===")
    print(f"CVs indexed: {total}")
    print(f"Categories: {categories}")
    print(f"Candidates ({len(known_names)}): {known_names[:10]}{'...' if len(known_names) > 10 else ''}")
    print()

    all_results: List[EvalResult] = []
    issue_count = 0

    for cat, questions in TEST_QUESTIONS.items():
        print(f"\n{'='*60}")
        print(f"  {cat.upper()} ({len(questions)} questions)")
        print(f"{'='*60}")
        for q in questions:
            r = evaluate_question(es, cat, q, known_names)
            all_results.append(r)
            status = "OK" if not r.issues else "ISSUE"
            if r.issues:
                issue_count += len(r.issues)

            print(f"\n[{status}] {q}")
            print(f"  intent={r.intent} | mode={r.route_mode} | docs={r.n_docs}")
            if r.topic:
                print(f"  topic={r.topic}")
            if r.mandatory:
                print(f"  mandatory={r.mandatory} | no_exact={r.no_exact_match}")
            if r.agg_intent:
                print(f"  agg={r.agg_intent} | stats={r.has_stats}")
            if r.doc_names:
                print(f"  top: {', '.join(r.doc_names)}")
            for iss in r.issues:
                print(f"  ⚠ {iss}")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(all_results)}")
    print(f"Total issues: {issue_count}")

    intent_failures = [r for r in all_results if any("INTENT" in i for i in r.issues)]
    retrieval_failures = [r for r in all_results if any("RETRIEVAL" in i or "ROUTE_ERROR" in i for i in r.issues)]
    comparison_failures = [r for r in all_results if any("COMPARISON" in i for i in r.issues)]

    print(f"\nIntent misrouting ({len(intent_failures)}):")
    for r in intent_failures:
        print(f"  - [{r.category}] {r.question}")

    print(f"\nRetrieval failures ({len(retrieval_failures)}):")
    for r in retrieval_failures:
        print(f"  - [{r.category}] {r.question}")

    print(f"\nComparison failures ({len(comparison_failures)}):")
    for r in comparison_failures:
        print(f"  - [{r.category}] {r.question}")

    # General chat intent check
    general = [r for r in all_results if r.category == "general_chat"]
    general_ok = sum(1 for r in general if r.intent != "cv")
    print(f"\nGeneral chat correctly routed: {general_ok}/{len(general)}")

    # Save JSON report
    report_path = os.path.join(ROOT, "scripts", "chatbot_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "category": r.category,
                    "question": r.question,
                    "intent": r.intent,
                    "route_mode": r.route_mode,
                    "n_docs": r.n_docs,
                    "doc_names": r.doc_names,
                    "no_exact_match": r.no_exact_match,
                    "issues": r.issues,
                }
                for r in all_results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
