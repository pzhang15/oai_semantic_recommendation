from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional


CANON_COLORS = [
    "black","white","gray","blue","navy","green","olive","beige","brown","red","pink","orange","yellow","purple"
]

CANON_MATERIALS = [
    "cotton","linen","wool","cashmere","silk","polyester","nylon","spandex","leather","vegan leather","denim","fleece","down"
]

CANON_CATEGORIES = [
    "t-shirt","shirt","camp shirt","polo","blouse","dress","jeans","pants","shorts","skirt","sneakers","boots","sandals","jacket","coat","sweater","hoodie","accessories"
]


def _lower_dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        y = str(x).strip().lower()
        if not y or y in seen:
            continue
        seen.add(y)
        out.append(y)
    return out


def normalize_occasion(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    mapping = {
        "beach": "beach",
        "wedding": "wedding",
        "office": "office",
        "work": "office",
        "gym": "gym",
        "party": "party",
        "travel": "travel",
        "commute": "travel",
        "rain": "rainy-day",
        "rainy": "rainy-day",
        "rainy day": "rainy-day",
        "cold": "cold-weather",
        "winter": "cold-weather",
        "hot": "hot-weather",
        "summer": "hot-weather",
        "outdoor": "outdoor",
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return t


def normalize_colors(colors: Iterable[str]) -> List[str]:
    out: List[str] = []
    for c in _lower_dedupe(colors):
        if c in CANON_COLORS:
            out.append(c)
            continue
        if any(x in c for x in ["sky blue", "light blue", "navy blue", "blueish"]):
            out.append("blue")
            continue
        if c in {"navy blue", "navy"}:
            out.append("navy")
            continue
        if c in {"tan", "beige", "khaki", "cream"}:
            out.append("beige")
            continue
        if c in {"ivory", "off-white", "off white"}:
            out.append("white")
            continue
        if c in {"charcoal", "light grey", "dark grey", "grey"}:
            out.append("gray")
            continue
        if c in {"camel"}:
            out.append("brown")
            continue
    return _lower_dedupe(out)


def normalize_materials(materials: Iterable[str]) -> List[str]:
    out: List[str] = []
    for m in _lower_dedupe(materials):
        if m in CANON_MATERIALS:
            out.append(m)
            continue
        if m in {"pleather", "pu leather", "faux leather"}:
            out.append("vegan leather")
            continue
        if m in {"elastane", "lycra"}:
            out.append("spandex")
            continue
        if m in {"down-filled", "puffer"}:
            out.append("down")
            continue
        if m.startswith("poly"):
            out.append("polyester")
            continue
    return _lower_dedupe(out)


_CATEGORY_MAP = {
    "tee": "t-shirt",
    "tshirt": "t-shirt",
    "t shirt": "t-shirt",
    "button-down": "shirt",
    "button down": "shirt",
    "oxford": "shirt",
    "chinos": "pants",
    "trousers": "pants",
    "sneaker": "sneakers",
    "trainers": "sneakers",
    "heels": "shoes",
    "boots": "boots",
    "sandals": "sandals",
    "hooded": "hoodie",
}


def normalize_categories(categories: Iterable[str]) -> List[str]:
    out: List[str] = []
    for c in _lower_dedupe(categories):
        if c in CANON_CATEGORIES:
            out.append(c)
            continue
        if c in _CATEGORY_MAP:
            out.append(_CATEGORY_MAP[c])
            continue
        # crude mapping
        if "shirt" in c and c not in out:
            out.append("shirt")
            continue
        if "jacket" in c and "jacket" not in out:
            out.append("jacket")
            continue
        if "coat" in c and "coat" not in out:
            out.append("coat")
            continue
        if "sweater" in c and "sweater" not in out:
            out.append("sweater")
            continue
    return _lower_dedupe(out)


class Budget:
    def __init__(self, min: Optional[float], max: Optional[float], currency: str) -> None:
        self.min = min
        self.max = max
        self.currency = currency


_CURR_BY_SYMBOL = {"$": "USD", "€": "EUR", "£": "GBP"}


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").strip())
    except Exception:
        return None


def extract_budget(text: str) -> Budget:
    t = (text or "").strip().lower()
    currency = None
    for sym, code in _CURR_BY_SYMBOL.items():
        if sym in t:
            currency = code
            break
    if currency is None:
        if " eur" in t or " euro" in t:
            currency = "EUR"
        elif " gbp" in t or " pound" in t:
            currency = "GBP"
        elif " usd" in t or "$" in t:
            currency = "USD"
        else:
            currency = "USD"

    # between A and B | A-B range
    m = re.search(r"between\s*([\d,]+)\s*(?:-|to|and)\s*([\d,]+)", t)
    if not m:
        m = re.search(r"([\d,]+)\s*[-–—]\s*([\d,]+)", t)
    if m:
        a = _parse_number(m.group(1))
        b = _parse_number(m.group(2))
        if a is not None and b is not None:
            lo, hi = (a, b) if a <= b else (b, a)
            return Budget(lo, hi, currency)

    # under / less than / < X
    m = re.search(r"(?:under|less than|<)\s*([\$€£]?\s*[\d,]+)", t)
    if m:
        x = m.group(1)
        x = x.replace("$", "").replace("€", "").replace("£", "")
        val = _parse_number(x)
        if val is not None:
            return Budget(None, val, currency)

    # around X / ~X
    m = re.search(r"(?:around|approx|~)\s*([\$€£]?\s*[\d,]+)", t)
    if m:
        x = m.group(1)
        x = x.replace("$", "").replace("€", "").replace("£", "")
        val = _parse_number(x)
        if val is not None:
            lo = round(val * 0.8, 2)
            hi = round(val * 1.2, 2)
            if lo > hi:
                lo, hi = hi, lo
            return Budget(lo, hi, currency)

    # plain under like "$120" alone
    m = re.search(r"([\$€£])\s*([\d,]+)", t)
    if m:
        sym = m.group(1)
        val = _parse_number(m.group(2))
        curr = _CURR_BY_SYMBOL.get(sym, currency)
        if val is not None:
            return Budget(None, val, curr)

    return Budget(None, None, currency)


