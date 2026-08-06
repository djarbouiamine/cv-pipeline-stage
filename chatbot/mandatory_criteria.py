"""Critères obligatoires dans une question : ne pas recommander sans preuve explicite."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MandatoryRequirement:
    label: str
    patterns: List[str]
    user_facing: str


def _haystack(doc: Dict) -> str:
    parts: List[str] = []
    for key in (
        "nom", "text", "categorie_principale", "certifications", "diplomes",
        "localisation",
    ):
        v = doc.get(key)
        if v:
            parts.append(str(v))
    for key in ("technologies", "langages", "frameworks", "langues", "projets", "description_projets"):
        v = doc.get(key) or []
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    return " ".join(parts).lower()


def _matches(haystack: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, haystack, re.IGNORECASE):
            return True
    return False


REQUIREMENT_RULES: List[MandatoryRequirement] = [
    MandatoryRequirement(
        "aws_cert",
        [r"aws\s+certif", r"certification\s+aws", r"aws\s+certified", r"certifi[eé]\s+aws"],
        "certification AWS explicite",
    ),
    MandatoryRequirement(
        "azure_cert",
        [r"azure\s+certif", r"certification\s+azure", r"certifi[eé]\s+azure"],
        "certification Azure explicite",
    ),
    MandatoryRequirement(
        "blockchain",
        [r"blockchain", r"ethereum", r"solidity", r"web3", r"smart\s+contract", r"hyperledger"],
        "expérience blockchain explicite",
    ),
    MandatoryRequirement(
        "rust",
        [r"\brust\b"],
        "compétence Rust explicite",
    ),
    MandatoryRequirement(
        "japanese",
        [r"japanese", r"japonais", r"\bnihongo\b"],
        "maîtrise du japonais explicite",
    ),
    MandatoryRequirement(
        "gcp",
        [r"\bgcp\b", r"google cloud platform", r"google cloud expert"],
        "expérience GCP explicite",
    ),
    MandatoryRequirement(
        "go",
        [r"\bgolang\b", r"\bgo developer\b", r"\bgo\b.*\bdeveloper\b"],
        "compétence Go explicite",
    ),
    MandatoryRequirement(
        "phd",
        [r"\bphd\b", r"\bph\.d\b", r"doctorat", r"doctoral"],
        "doctorat / PhD explicite",
    ),
    MandatoryRequirement(
        "azure_skill",
        [r"azure expert", r"\bazure\b.*\b(expert|architect|engineer)\b"],
        "expertise Azure explicite",
    ),
    MandatoryRequirement(
        "aws_skill",
        [r"\baws\b", r"amazon web services"],
        "expérience AWS explicite",
    ),
]


def detect_mandatory_requirements(question: str) -> List[MandatoryRequirement]:
    q = question.lower()
    found: List[MandatoryRequirement] = []
    for rule in REQUIREMENT_RULES:
        if not _matches(q, rule.patterns):
            continue
        if rule.label == "aws_skill" and any(r.label == "aws_cert" for r in found):
            continue
        if rule.label == "azure_skill" and any(r.label == "azure_cert" for r in found):
            continue
        found.append(rule)
    return found


def split_docs_by_requirements(
    docs: List[Dict],
    requirements: List[MandatoryRequirement],
) -> Tuple[List[Dict], List[Dict]]:
    """Docs qui satisfont TOUS les critères obligatoires vs les autres (proximité)."""
    if not requirements:
        return docs, []

    matching: List[Dict] = []
    others: List[Dict] = []
    for doc in docs:
        h = _haystack(doc)
        ok = all(_matches(h, req.patterns) for req in requirements)
        if ok:
            matching.append(doc)
        else:
            others.append(doc)
    return matching, others


def partial_match_score(doc: Dict, requirements: List[MandatoryRequirement]) -> int:
    if not requirements:
        return 0
    h = _haystack(doc)
    return sum(1 for req in requirements if _matches(h, req.patterns))


def requirements_summary(requirements: List[MandatoryRequirement]) -> str:
    if not requirements:
        return ""
    return ", ".join(r.user_facing for r in requirements)
