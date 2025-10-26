from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import BadRequestError, OpenAI

from src.core.config import get_settings
from src.core.filters import apply_filters
from src.core.rerank import judge_rerank
from src.core.retriever import load_products_table
from src.models.outfit import OutfitItem, OutfitPlan, build_outfit_schema
from src.models.normalize import CANON_COLORS, CANON_MATERIALS
from src.telemetry.timer import span
from src.telemetry.counters import add_usage


SLOT_MAP = {
    "top": ["t-shirt", "shirt", "blouse", "polo", "sweater", "hoodie", "camp shirt"],
    "bottom": ["jeans", "pants", "shorts", "skirt"],
    "shoes": ["sneakers", "boots", "sandals", "loafers"],
    "outerwear": ["jacket", "coat", "parka", "raincoat", "rain coat"],
    "accessories": ["belt", "hat", "cap", "scarf", "bag", "backpack", "sunglasses"],
}


def categorize_candidate(title: str) -> Optional[str]:
    t = (title or "").lower()
    for slot, keys in SLOT_MAP.items():
        if any(k in t for k in keys):
            return slot
    return None


def guess_color(title: str) -> Optional[str]:
    t = (title or "").lower()
    for c in ["navy"] + CANON_COLORS:
        if c in t:
            return c
    if "beige" in t or "khaki" in t or "tan" in t:
        return "beige"
    return None


def guess_material(title: str) -> Optional[str]:
    t = (title or "").lower()
    for m in CANON_MATERIALS:
        if m in t:
            return m
    if "faux leather" in t or "vegan leather" in t:
        return "vegan leather"
    if re.search(r"\bpoly\b", t):
        return "polyester"
    if "elastane" in t or "lycra" in t:
        return "spandex"
    return None


def prepare_outfit_pool(facets, candidates: List[Dict[str, Any]], pool_size: int) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for it in candidates[: pool_size]:
        pid = it.get("id")
        title = it.get("title")
        if not pid or not title:
            continue
        slot_guess = categorize_candidate(title)
        color_guess = guess_color(title)
        material_guess = guess_material(title)
        pool.append({
            "id": pid,
            "title": title,
            "brand": it.get("brand"),
            "price": it.get("price"),
            "slot_guess": slot_guess,
            "color_guess": color_guess,
            "material_guess": material_guess,
        })
    # Bias pool toward priced items to guide LLM selection
    for p in pool:
        try:
            p["price_num"] = float(p["price"]) if p.get("price") not in (None, "", "None") else None
        except Exception:
            p["price_num"] = None
    pool.sort(key=lambda x: (x.get("price_num") is None,))  # stable: priced first
    return pool


def _call_chat_completion(client: OpenAI, model_id: str, messages: list[dict], schema: Optional[dict], max_tokens: int, timeout_secs: int):
    def _do(include_temp: bool, rf: Optional[dict]):
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "timeout": timeout_secs,
        }
        if include_temp:
            kwargs["temperature"] = 0
        if rf is not None:
            kwargs["response_format"] = rf
        return client.chat.completions.create(**kwargs)

    if schema is not None:
        try:
            return _do(True, schema)
        except BadRequestError:
            try:
                return _do(False, schema)
            except Exception:
                pass
        except Exception:
            pass
        try:
            return _do(False, {"type": "json_object"})
        except Exception:
            return _do(False, None)
    else:
        try:
            return _do(True, None)
        except BadRequestError:
            return _do(False, None)


def compose_outfit(facets, pool: List[Dict[str, Any]], settings) -> OutfitPlan:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    schema = build_outfit_schema()
    sys_prompt = (
        "Compose a coherent outfit from the candidate pool. Fill required slots first; ensure color/material harmony; "
        "avoid conflicts (e.g., hiking boots with formal dress). Prefer within budget. Do not reuse the same id across slots. "
        "Return ONLY JSON per schema."
    )
    facets_summary = {
        "occasion": facets.occasion,
        "season": facets.season,
        "budget": {"min": facets.budget_usd.min, "max": facets.budget_usd.max, "currency": facets.budget_usd.currency},
        "colors": facets.colors,
        "materials": facets.materials,
        "must_have": facets.must_have,
        "hard_constraints": facets.hard_constraints,
        "required_slots": get_settings().outfit_required_slots,
    }
    pool_min = pool[:]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Facets:" + json.dumps(facets_summary, separators=(",", ":")) + "\nPool:" + json.dumps(pool_min, separators=(",", ":"))},
    ]

    attempt = 0
    last_err = None
    while attempt <= settings.outfit_retries:
        try:
            attempt += 1
            if last_err:
                messages[-1] = {"role": "user", "content": messages[-1]["content"] + f"\nNote: Fix previous issue: {last_err}"}
    with span("outfit") as sp:
        resp = _call_chat_completion(client, settings.outfit_model, messages, schema, settings.outfit_max_tokens, settings.outfit_timeout_secs)
            content = resp.choices[0].message.content or "{}"
    try:
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        tin = tout = 0
    add_usage("outfit", settings.outfit_model, tin, tout, sp.ms)
            data = json.loads(content)
            plan = OutfitPlan.model_validate(data)
            return plan
        except Exception as e:
            last_err = str(e)
            if attempt > settings.outfit_retries:
                raise
    raise RuntimeError("compose_outfit failed")


def _canonical_price(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _pick_from_pool_for_slot(slot: str, pool: List[dict], used_ids: set, max_items: int) -> List[dict]:
    candidates = [c for c in pool if c.get("id") and c["id"] not in used_ids]
    def score(c: dict) -> tuple[int, int]:
        slot_match = 0 if c.get("slot_guess") == slot else 1
        price_missing = 1 if _canonical_price(c.get("price")) is None else 0
        return (slot_match, price_missing)
    candidates.sort(key=score)
    picked: List[dict] = []
    for c in candidates:
        picked.append(c)
        used_ids.add(c["id"])
        if len(picked) >= max_items:
            break
    return picked


def validate_and_finalize(plan: OutfitPlan, pool: List[Dict[str, Any]], facets, settings) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    required = settings.outfit_required_slots
    pool_by_id = {str(x["id"]): x for x in pool}
    # Ensure required slots present
    if plan.slots is None:
        plan.slots = {}
    for slot in required:
        plan.slots.setdefault(slot, [])

    # Enforce unique ids across slots and clamp alternates
    used_ids: set = set()
    for slot, items in list(plan.slots.items()):
        cleaned: List[OutfitItem] = []
        for it in items:
            pid = getattr(it, "id", None)
            if not pid or pid in used_ids:
                continue
            cleaned.append(it)
            used_ids.add(pid)
            if len(cleaned) >= 1 + settings.outfit_max_alternates:
                break
        plan.slots[slot] = cleaned

    # Fill empty required slots from pool
    for slot in required:
        if not plan.slots.get(slot):
            fallback = _pick_from_pool_for_slot(slot, pool, used_ids, max_items=1 + settings.outfit_max_alternates)
            plan.slots[slot] = [OutfitItem(id=str(p["id"]), slot=slot, title=p.get("title") or "", brand=p.get("brand"), price=_canonical_price(p.get("price"))) for p in fallback]

    # Rebuild items from products table to ensure canonical fields
    df = load_products_table(get_settings().parquet_path).set_index("id")

    total = 0.0
    has_unknown = False
    palette: List[str] = []
    materials: List[str] = []

    slots_out: Dict[str, List[Dict[str, Any]]] = {}
    for slot, items in plan.slots.items():
        # Convert OutfitItem to dicts with canonical joins
        enriched: List[Dict[str, Any]] = []
        for it in items:
            pid = str(it.id)
            base_title = it.title
            base_brand = it.brand
            base_price = _canonical_price(it.price)
            image_url = None
            product_url = None
            if pid in df.index:
                row = df.loc[pid]
                base_title = base_title or row.get("title")
                base_brand = base_brand or row.get("brand")
                if base_price is None and isinstance(row.get("price"), (int, float)):
                    base_price = _canonical_price(row.get("price"))
                image_url = row.get("image_url")
                product_url = row.get("product_url")
            enriched.append({
                "id": pid,
                "title": base_title,
                "brand": base_brand,
                "price": base_price,
                "image_url": image_url,
                "product_url": product_url,
                "why": it.why,
                "score": None,
            })

        # If first choice has no price, try to swap with a priced alternate or pool candidate
        if enriched and enriched[0]["price"] is None:
            swapped = False
            for k in range(1, len(enriched)):
                if enriched[k]["price"] is not None:
                    enriched[0], enriched[k] = enriched[k], enriched[0]
                    swapped = True
                    break
            if not swapped:
                extras = _pick_from_pool_for_slot(slot, pool, used_ids=set(), max_items=5)
                existing_ids = {e["id"] for e in enriched}
                for cand in extras:
                    if cand["id"] in existing_ids:
                        continue
                    pid = cand["id"]
                    price = _canonical_price(cand.get("price"))
                    if pid in df.index:
                        row = df.loc[pid]
                        if price is None and isinstance(row.get("price"), (int, float)):
                            price = _canonical_price(row.get("price"))
                        title = cand.get("title") or row.get("title")
                        brand = cand.get("brand") or row.get("brand")
                        image_url = cand.get("image_url") or row.get("image_url")
                        product_url = cand.get("product_url") or row.get("product_url")
                    else:
                        title = cand.get("title")
                        brand = cand.get("brand")
                        image_url = cand.get("image_url")
                        product_url = cand.get("product_url")
                    if price is not None:
                        enriched.insert(0, {
                            "id": pid,
                            "title": title,
                            "brand": brand,
                            "price": price,
                            "image_url": image_url,
                            "product_url": product_url,
                            "why": cand.get("why"),
                            "score": cand.get("score"),
                        })
                        break
        # Clamp alternates
        if len(enriched) > 1 + settings.outfit_max_alternates:
            enriched = enriched[: 1 + settings.outfit_max_alternates]

        # Accumulate palette/materials from pool guesses
        for e in enriched:
            p = pool_by_id.get(e["id"]) or {}
            if p.get("color_guess"):
                palette.append(p.get("color_guess"))
            if p.get("material_guess"):
                materials.append(p.get("material_guess"))
        slots_out[slot] = enriched

    # Compute numeric total
    for slot in required:
        first = (slots_out.get(slot) or [])[:1]
        if not first:
            continue
        pr = first[0].get("price")
        if pr is None:
            has_unknown = True
            pr = 0.0
        total += float(pr)

    bud_max = facets.budget_usd.max if facets and facets.budget_usd else None
    tol = settings.outfit_price_tolerance
    under_budget = True
    if isinstance(bud_max, (int, float)):
        under_budget = total <= bud_max * (1.0 + tol)

    def top_k(values: List[str], k: int = 2) -> List[str]:
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return [kv[0] for kv in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k] if kv[0]]

    palette_final = top_k(palette, 2)
    materials_final = top_k(materials, 3)

    summary = "Coherent outfit composed across slots."
    if facets.occasion:
        summary = f"{facets.occasion.capitalize()} outfit assembled."
    tips = "Consider tailoring fit and adding weather-appropriate layers."

    resp = {
        "slots": slots_out,
        "total_price": round(total, 2),
        "under_budget": bool(under_budget),
        "summary": summary,
        "tips": tips,
    }
    trace = {
        "palette": palette_final,
        "materials": materials_final,
        "total_price_before_adjust": round(total, 2),
        "has_unknown_prices": has_unknown,
    }
    return resp, trace


