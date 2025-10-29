from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re

from openai import OpenAI, BadRequestError

from src.core.config import get_settings
from src.models.schemas import QueryFacets, build_structured_output_schema
from src.models.normalize import (
    normalize_occasion,
    normalize_colors,
    normalize_materials,
    normalize_categories,
    extract_budget,
)
from src.telemetry.timer import span
from src.telemetry.counters import add_usage
from src.core.clients import get_openai_client


class ParserError(Exception):
    pass


SYSTEM_PROMPT = (
    "You extract shopping facets from a user query for fashion recommendations. "
    "Return ONLY JSON that strictly matches the provided JSON Schema. "
    "Be conservative: do not infer gender unless explicitly stated (e.g., 'men', 'women', 'unisex'). "
    "If uncertain, leave optional fields empty or []. "
    "Use the user's budget hints like 'under $120', ranges, or currencies. "
    "Extract must_have vs hard_constraints (negations like 'no logos', 'avoid X'). "
)


def _post_normalize(facets: QueryFacets, original_text: str) -> QueryFacets:
    # Normalize lists and canonical mappings
    facets.style_adjectives = sorted(set([s.strip().lower() for s in facets.style_adjectives if s]))
    facets.colors = normalize_colors(facets.colors)
    facets.materials = normalize_materials(facets.materials)
    facets.categories_include = normalize_categories(facets.categories_include)
    facets.categories_exclude = normalize_categories(facets.categories_exclude)
    facets.occasion = normalize_occasion(facets.occasion)
    # Clamp num_items handled by validator, ensure lists are lists
    facets.must_have = list({s.strip().lower() for s in (facets.must_have or []) if s})
    facets.hard_constraints = list({s.strip().lower() for s in (facets.hard_constraints or []) if s})

    # Heuristic: add materials explicitly mentioned in text
    mats = _extract_materials(original_text)
    if mats:
        facets.materials = normalize_materials(list({*facets.materials, *mats}))

    # Heuristic: extract must-have features from text (e.g., waterproof, pockets)
    musts = _extract_must_haves(original_text)
    if musts:
        facets.must_have = sorted(list({*(facets.must_have or []), *musts}))

    # Heuristic: extract categories from text
    cats = _extract_categories(original_text)
    if cats:
        facets.categories_include = normalize_categories(list({*(facets.categories_include or []), *cats}))

    # Heuristic: explicit gender/fit and season if present in text
    gender, size_note = _extract_gender_and_size(original_text)
    if gender and (facets.gender_or_fit is None or facets.gender_or_fit == "unknown"):
        facets.gender_or_fit = gender
    if size_note and not facets.size_notes:
        facets.size_notes = size_note
    if not facets.season:
        season = _extract_season(original_text)
        if season:
            facets.season = season

    # Heuristic: occasion from text if model omitted it
    if not facets.occasion:
        occ = _extract_occasion(original_text)
        if occ:
            facets.occasion = occ

    # Budget extraction as fallback
    if (facets.budget_usd.min is None and facets.budget_usd.max is None) or not facets.budget_usd.currency:
        b = extract_budget(original_text)
        facets.budget_usd.min = b.min
        facets.budget_usd.max = b.max
        facets.budget_usd.currency = b.currency or facets.budget_usd.currency or "USD"

    # Basic sanity: swap if min>max
    mi = facets.budget_usd.min
    ma = facets.budget_usd.max
    if isinstance(mi, (int, float)) and isinstance(ma, (int, float)) and mi > ma:
        facets.budget_usd.min, facets.budget_usd.max = ma, mi

    # Heuristic: ensure explicit negations from text are captured in hard_constraints
    negs = _extract_negations(original_text)
    if negs:
        existing = set(facets.hard_constraints or [])
        for n in negs:
            if n not in existing:
                facets.hard_constraints.append(n)
        facets.hard_constraints = list({s.strip().lower() for s in facets.hard_constraints if s})

    # Intent fallback if model left it unknown
    if (facets.intent or "unknown") == "unknown":
        inferred = _infer_intent(original_text)
        if inferred:
            facets.intent = inferred

    return facets


_CACHE: Dict[Tuple[str, str], Tuple[QueryFacets, float]] = {}
_CACHE_MAX = 128


def _cache_get(key: Tuple[str, str]) -> Optional[QueryFacets]:
    item = _CACHE.get(key)
    if not item:
        return None
    return item[0]


def _cache_put(key: Tuple[str, str], value: QueryFacets) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # drop oldest arbitrary
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (value, time.time())


def _call_chat_completion(client: OpenAI, model_id: str, messages: list[dict], schema: dict, max_tokens: int, timeout_secs: int) -> Any:
    # Helper to conditionally include temperature
    def _do_call(response_format: dict | None, include_temp: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if include_temp:
            kwargs["temperature"] = 0
        # Prefer per-request timeout
        kwargs["timeout"] = timeout_secs
        return client.chat.completions.create(**kwargs)

    # 1) Try schema with temperature=0
    try:
        return _do_call(schema, include_temp=True)
    except BadRequestError as e:
        # If model rejects temperature, retry without it
        msg = str(e)
        if "param': 'temperature" in msg or "Unsupported value" in msg or "temperature" in msg:
            try:
                return _do_call(schema, include_temp=False)
            except Exception:
                pass
        # Fall through to other formats
    except Exception:
        pass

    # 2) Fallback: json_object without temperature
    try:
        return _do_call({"type": "json_object"}, include_temp=False)
    except Exception:
        pass

    # 3) Plain completion without response_format, no temperature
    return _do_call(None, include_temp=False)


def _extract_negations(text: str) -> list[str]:
    # Capture phrases like "no logos", "avoid polyester", "without heels", "not leather"
    t = (text or "").strip().lower()
    results: list[str] = []
    # common patterns
    patterns = [
        r"\bno\s+([a-z][a-z\-\s]{1,30})",
        r"\bavoid\s+([a-z][a-z\-\s]{1,30})",
        r"\bwithout\s+([a-z][a-z\-\s]{1,30})",
        r"\bnot\s+([a-z][a-z\-\s]{1,30})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            phrase = m.group(1).strip()
            # collapse multi-space and limit to first two words to avoid over-capture
            words = [w for w in phrase.split() if w not in {"any", "a", "the"}]
            if not words:
                continue
            normalized = " ".join(words[:3])
            # special-case: skip 'vegan leather' as negation (it's a positive material)
            if normalized.startswith("vegan leather"):
                continue
            # store either the object or full phrase like 'no logos'
            if pat.startswith(r"\bno"):
                results.append(f"no {normalized}")
            elif pat.startswith(r"\bavoid"):
                results.append(f"avoid {normalized}")
            elif pat.startswith(r"\bwithout"):
                results.append(f"without {normalized}")
            else:
                results.append(f"not {normalized}")
    # dedupe
    out = []
    seen = set()
    for s in results:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_materials(text: str) -> list[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    # multi-word first
    if "vegan leather" in t or "faux leather" in t or "pu leather" in t:
        found.add("vegan leather")
    # singles
    keywords = [
        "linen","cotton","wool","cashmere","silk","polyester","nylon","spandex","leather","denim","fleece","down"
    ]
    # common synonyms
    if "elastane" in t or "lycra" in t:
        found.add("spandex")
    if re.search(r"\bpoly\b", t):
        found.add("polyester")
    for k in keywords:
        if k in t:
            found.add(k)
    return sorted(found)


def _extract_gender_and_size(text: str) -> tuple[Optional[str], Optional[str]]:
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

    # Basic size extraction (keep simple): tokens like "\b[XS|S|M|L|XL|XXL]\b", numeric shoes, or waist formats
    m = re.search(r"\b(XXL|XL|XS|S|M|L)\b", t)
    if m:
        size = m.group(1)
    # shoe sizes
    if size is None:
        m = re.search(r"\bUS\s*([0-9]{1,2}(?:\.[0-9])?)\b", text, re.IGNORECASE)
        if m:
            size = f"US {m.group(1)}"
    # waist notation like 32x32 or 32 waist
    if size is None:
        m = re.search(r"\b([2-5][0-9])\s*(?:x|×|waist)\s*([2-5][0-9])?\b", t)
        if m:
            size = m.group(0)
    return gender, size


def _extract_season(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if "winter" in t:
        return "winter"
    if "summer" in t:
        return "summer"
    if "spring" in t:
        return "spring"
    if "fall" in t or "autumn" in t:
        return "fall"
    return None


def _extract_occasion(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    # Order matters: more specific first
    checks: list[tuple[list[str], str]] = [
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


def _extract_must_haves(text: str) -> list[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    # map synonyms to canonical keys
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
    }
    for canon, keys in synonyms.items():
        if any(k in t for k in keys):
            found.add(canon)
    # Normalize water-resistant to waterproof if rainy context present
    if "water-resistant" in found and ("rain" in t or "rainy" in t or "commute" in t):
        found.add("waterproof")
    return sorted(found)


def _extract_categories(text: str) -> list[str]:
    t = (text or "").strip().lower()
    found: set[str] = set()
    # direct matches for canonical categories and common synonyms
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
        "sneakers": ["sneakers", "sneaker", "trainers"],
        "boots": ["boots", "boot"],
        "sandals": ["sandals", "sandal"],
        "jacket": ["jacket"],
        "coat": ["coat", "parka"],
        "sweater": ["sweater", "knit"],
        "hoodie": ["hoodie", "hooded"],
        "accessories": ["belt", "hat", "cap", "scarf", "gloves"],
    }
    for canon, keys in mapping.items():
        if any(k in t for k in keys):
            found.add(canon)
    return sorted(found)


def _infer_intent(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    # Compare intent
    if any(k in t for k in ["compare", " vs ", "versus", "difference between", "compare to"]):
        return "compare"
    # Outfit intent
    if any(k in t for k in ["outfit", "look", "ensemble", "set"]):
        return "outfit"
    # Category mention → search intent
    if _extract_categories(t):
        return "search"
    # Occasion mention also likely a search/outfit; prefer search as conservative default
    if _extract_occasion(t):
        return "search"
    return None


def parse_query(text: str, *, model: Optional[str] = None, retries: int | None = None) -> QueryFacets:
    settings = get_settings()
    query = " ".join((text or "").split())
    if not query:
        raise ValueError("Query is empty after normalization")

    model_id = model or settings.model_parser
    max_retries = settings.parser_retries if retries is None else int(retries)

    cache_key = (query, model_id)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    client = get_openai_client()

    schema = build_structured_output_schema(QueryFacets, name="query_facets")

    attempt = 0
    last_err: Optional[str] = None
    while attempt <= max_retries:
        attempt += 1
        user_prompt = query
        if last_err:
            user_prompt = f"{query}\n\nNote: Fix previous validation error: {last_err}."
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            with span("parser") as sp:
                resp = _call_chat_completion(
                    client, model_id, messages, schema, settings.parser_max_tokens, settings.parser_timeout_secs
                )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            facets = QueryFacets.model_validate(data)
            facets = _post_normalize(facets, query)
            # usage tokens
            try:
                usage = getattr(resp, "usage", None)
                tin = int(getattr(usage, "prompt_tokens", 0) or 0)
                tout = int(getattr(usage, "completion_tokens", 0) or 0)
            except Exception:
                tin = tout = 0
            add_usage("parser", model_id, tin, tout, sp.ms)
            _cache_put(cache_key, facets)
            return facets
        except Exception as e:
            last_err = str(e)
            if attempt > max_retries:
                raise ParserError(f"Parser failed: {last_err}")

    # Should not reach
    raise ParserError("Parser failed: unknown error")


def to_public_dict(facets: QueryFacets) -> Dict[str, Any]:
    return facets.model_dump()


