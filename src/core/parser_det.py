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


def _extract_materials(text: str) -> List[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    if "vegan leather" in t or "faux leather" in t or "pu leather" in t:
        found.add("vegan leather")
    keywords = [
        "linen","cotton","wool","cashmere","silk","polyester","nylon","spandex","leather","denim","fleece","down"
    ]
    if "elastane" in t or "lycra" in t:
        found.add("spandex")
    if re.search(r"\bpoly\b", t):
        found.add("polyester")
    for k in keywords:
        if k in t:
            found.add(k)
    return sorted(found)


def _extract_categories(text: str) -> List[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    mapping = {
        "t-shirt": ["t-shirt", "tshirt", "tee", "t shirt"],
        "shirt": ["shirt", "button-down", "button down", "oxford"],
        "polo": ["polo"],
        "blouse": ["blouse"],
        "dress": ["dress"],
        "jeans": ["jeans"],
        "pants": ["pants", "trousers", "chinos"],
        "shorts": ["shorts"],
        "skirt": ["skirt"],
        "sneakers": ["sneakers", "sneaker", "trainers", "running shoes"],
        "boots": ["boots", "boot", "chelsea boots"],
        "sandals": ["sandals", "sandal"],
        "jacket": ["jacket"],
        "coat": ["coat", "parka", "puffer"],
        "sweater": ["sweater", "knit"],
        "hoodie": ["hoodie", "hooded"],
        "accessories": ["belt", "hat", "cap", "scarf", "gloves"],
        "handbag": ["handbag", "bag", "purse"],
        "backpack": ["backpack", "pack"],
    }
    for canon, keys in mapping.items():
        if any(k in t for k in keys):
            found.add(canon)
    return sorted(found)


def _extract_gender_and_size(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip().lower()
    gender: Optional[str] = None
    size: Optional[str] = None
    if re.search(r"\bmen\b|men's|mens\b", t):
        gender = "men"
    elif re.search(r"\bwomen\b|women's|womens\b|ladies\b", t):
        gender = "women"
    elif re.search(r"\bunisex\b", t):
        gender = "unisex"
    elif re.search(r"\bgirls\b", t):
        gender = "girls"
    elif re.search(r"\bboys\b", t):
        gender = "boys"
    m = re.search(r"\b(XXL|XL|XS|S|M|L)\b", t)
    if m:
        size = m.group(1)
    if size is None:
        m = re.search(r"\bUS\s*([0-9]{1,2}(?:\.[0-9])?)\b", text, re.IGNORECASE)
        if m:
            size = f"US {m.group(1)}"
    if size is None:
        m = re.search(r"\b([2-5][0-9])\s*(?:x|×|waist)\s*([2-5][0-9])?\b", t)
        if m:
            size = m.group(0)
    return gender, size


def _extract_occasion(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    checks: List[Tuple[List[str], str]] = [
        (["gym", "workout", "training"], "gym"),
        (["wedding"], "wedding"),
        (["beach"], "beach"),
        (["office", "work"], "office"),
        (["party"], "party"),
        (["travel", "trip", "commute"], "travel"),
        (["rain", "rainy"], "rainy-day"),
        (["outdoor", "hike", "hiking"], "outdoor"),
        (["cold", "winter"], "cold-weather"),
        (["hot", "summer"], "hot-weather"),
    ]
    for keywords, label in checks:
        if any(k in t for k in keywords):
            return label
    return None


def _extract_must_haves(text: str) -> List[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    synonyms = {
        "waterproof": ["waterproof", "water proof", "rain-proof", "rainproof"],
        "water-resistant": ["water resistant", "water-resistant", "water resist"],
        "breathable": ["breathable", "well-ventilated", "vented"],
        "pockets": ["pockets", "with pockets"],
        "hood": ["hood", "hooded"],
        "lightweight": ["lightweight", "light weight", "light-weight"],
        "insulated": ["insulated", "warm", "padded"],
        "quick-dry": ["quick dry", "quick-dry", "fast drying"],
        "stretchy": ["stretchy", "stretch", "elastic"],
        "packable": ["packable", "packs small"],
        "slip resistant": ["slip resistant", "non slip", "anti slip"],
    }
    for canon, keys in synonyms.items():
        if any(k in t for k in keys):
            found.add(canon)
    if "water-resistant" in found and ("rain" in t or "rainy" in t or "commute" in t):
        found.add("waterproof")
    return sorted(found)


def parse_deterministic(text: str) -> DetParseResult:
    t0 = time.perf_counter()
    q = " ".join((text or "").split())
    if not q:
        return DetParseResult(QueryFacets(), 0.0, {"low_info": True}, {}, 0.0)

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


