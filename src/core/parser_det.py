from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.models.normalize import (
    normalize_categories,
    normalize_colors,
    normalize_materials,
    extract_budget,
)
from src.models.schemas import QueryFacets


@dataclass
class DetParseResult:
    facets: QueryFacets
    confidence_overall: float
    flags: Dict[str, bool]
    matches: Dict[str, List[str]]
    ms: float


_COLOR_HINTS = [
    # Canonical colors and common variants
    "black","white","gray","grey","blue","navy","green","olive","beige","tan","brown",
    "red","pink","orange","yellow","purple","cream","ivory","off white","off-white","charcoal",
    "sky blue","light blue","navy blue","royal blue",
]


def _find_colors(text: str) -> List[str]:
    t = (text or "").strip().lower()
    hits: List[str] = []
    for w in _COLOR_HINTS:
        if w in t:
            hits.append(w)
    return normalize_colors(hits)


def _detect_ambiguity(text: str, cats: List[str], colors: List[str]) -> Dict[str, bool]:
    t = (text or "").strip().lower()
    flags: Dict[str, bool] = {}
    # Ambiguous color words without strong anchor
    if any(k in t for k in ["royal", "nude", "cream"]) and not colors:
        flags["ambiguous_color"] = True
    # Multiple distinct categories
    if len(cats) >= 2:
        flags["ambiguous_category"] = True
    # Low-info adjectives only
    if not cats and not colors and not re.search(r"\b(\$|under|over|less than|more than|between|range)\b", t):
        # contains only generic style words?
        if any(k in t for k in ["nice", "cute", "elegant", "cool", "stylish", "trendy"]):
            flags["low_info"] = True
    return flags


def _extract_negation_phrases(text: str) -> List[str]:
    t = (text or "").strip().lower()
    phrases: List[str] = []
    for pat in [r"\bno\s+([a-z][a-z\-\s]{1,30})", r"\bwithout\s+([a-z][a-z\-\s]{1,30})", r"\bavoid\s+([a-z][a-z\-\s]{1,30})", r"\bnot\s+([a-z][a-z\-\s]{1,30})"]:
        for m in re.finditer(pat, t):
            phrases.append(m.group(0).strip())
    # dedupe simple
    out: List[str] = []
    seen = set()
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_deterministic(text: str) -> DetParseResult:
    t0 = time.perf_counter()
    q = " ".join((text or "").split())
    if not q:
        return DetParseResult(QueryFacets(), 0.0, {"low_info": True}, {}, 0.0)

    from src.core.parser import (
        _extract_materials,
        _extract_categories,
        _extract_gender_and_size,
        _extract_occasion,
        _extract_must_haves,
    )

    # Extract raw candidates
    cats = _extract_categories(q)
    mats = _extract_materials(q)
    cols = _find_colors(q)
    gender, size_note = _extract_gender_and_size(q)
    occ = _extract_occasion(q)
    musts = _extract_must_haves(q)
    neg_phrases = _extract_negation_phrases(q)
    budget = extract_budget(q)

    # Canonicalize
    cats = normalize_categories(cats)
    mats = normalize_materials(mats)
    cols = normalize_colors(cols)

    # Negation handling: if "not leather" present, drop leather from positives
    hard_constraints = list(sorted(set(neg_phrases)))
    if any("leather" in p for p in hard_constraints):
        mats = [m for m in mats if m != "leather"]

    # Populate facets
    facets = QueryFacets()
    facets.categories_include = cats
    facets.colors = cols
    facets.materials = mats
    facets.gender_or_fit = gender or facets.gender_or_fit
    facets.size_notes = size_note or facets.size_notes
    facets.occasion = occ or facets.occasion
    facets.must_have = musts or []
    facets.hard_constraints = hard_constraints or []
    # Budget
    facets.budget_usd.min = budget.min
    facets.budget_usd.max = budget.max
    facets.budget_usd.currency = budget.currency or (facets.budget_usd.currency or "USD")

    # Confidence scoring
    conf = 0.0
    matches: Dict[str, List[str]] = {}
    if cats:
        conf += 0.25
        matches["categories"] = cats
    if facets.budget_usd.min is not None and facets.budget_usd.max is not None:
        conf += 0.3
    elif facets.budget_usd.min is not None or facets.budget_usd.max is not None:
        conf += 0.2
    if cols:
        conf += 0.1
        matches["colors"] = cols
    if mats:
        conf += 0.1
        matches["materials"] = mats
    if musts:
        conf += 0.05
        matches["must_have"] = musts

    # Ambiguity flags
    flags = _detect_ambiguity(q, cats, cols)

    # Contradictions: if negation conflicts with positives
    if any("not leather" in p or "without leather" in p for p in hard_constraints) and ("leather" in mats):
        conf -= 0.2

    # Clamp conf
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    t1 = time.perf_counter()
    return DetParseResult(facets=facets, confidence_overall=conf, flags=flags, matches=matches, ms=(t1 - t0) * 1000.0)


