"""Score multi-critères pour recommandations (délègue au classement final)."""

from __future__ import annotations

from typing import Dict, List

from chatbot.final_ranking import rerank_docs_by_job_relevance

RECOMMEND_HINTS = [
    "recommend", "recommande", "recommandation", "conseille", "embauche", "hire",
    "intern", "stage", "junior", "startup", "meilleur choix", "who would you",
]


def is_recommendation_question(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in RECOMMEND_HINTS) or "pour un" in q or "pour une" in q


def rerank_docs_for_recommendation(question: str, docs: List[Dict]) -> List[Dict]:
    return rerank_docs_by_job_relevance(question, docs)
