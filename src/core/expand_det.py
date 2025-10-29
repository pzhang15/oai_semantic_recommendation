from __future__ import annotations

from typing import List, Tuple


_ALIASES = {
    # categories/terms
    "trainers": "sneakers",
    "t-shirt": "tshirt",
    "tee": "tshirt",
    "pleather": "faux leather",
    "gore-tex": "gore tex",
    "rain proof": "waterproof",
    "rainproof": "waterproof",
    "non slip": "slip resistant",
    "anti slip": "slip resistant",
    "colour": "color",
    "jumper": "sweater",
    "waistcoat": "vest",
    "burgundy": "maroon",
    "khaki": "tan",
}

_BRAND_ALIASES = {
    "docs": "dr martens",
    "doc martens": "dr martens",
    "tnf": "the north face",
}

_TAXONOMY_PARENTS = {
    "chelsea boots": "boots",
    "running shoes": "sneakers",
}


def _try_replace(q: str, old: str, new: str) -> str | None:
    t = q.lower()
    if old in t and new != old:
        return t.replace(old, new)
    return None


def _singularize_basic(q: str) -> str | None:
    t = q.lower().strip()
    if "dresses" in t:
        return t.replace("dresses", "dress")
    if "shirts" in t:
        return t.replace("shirts", "shirt")
    if "jeans" in t:
        return t  # jeans is plural form commonly
    if t.endswith("s ") or t.endswith("s"):
        # avoid numbers/units
        return t[:-1]
    return None


def generate_deterministic_expansions(query: str, max_items: int = 2) -> List[str]:
    q = " ".join((query or "").split()).strip()
    if not q:
        return []
    cands: List[str] = []
    # 1) Brand alias
    for k, v in _BRAND_ALIASES.items():
        r = _try_replace(q, k, v)
        if r and r != q:
            cands.append(r)
            break
    # 2) Taxonomy parent
    for k, v in _TAXONOMY_PARENTS.items():
        r = _try_replace(q, k, v)
        if r and r != q:
            cands.append(r)
            break
    # 3) Alias/synonym swaps
    for k, v in _ALIASES.items():
        r = _try_replace(q, k, v)
        if r and r != q:
            cands.append(r)
            if len(cands) >= max_items:
                break
    # 4) Singularization as last resort
    if len(cands) < max_items:
        r = _singularize_basic(q)
        if r and r != q:
            cands.append(r)

    # Dedupe while preserving order and cut to max_items
    seen: set[str] = set()
    out: List[str] = []
    for s in cands:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_items:
            break
    return out


