from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.core.config import get_settings
from src.core.parser import ParserError, parse_query
from src.core.retriever import retrieve
from src.core.filters import apply_filters
from src.core.rerank import judge_rerank
from src.core.outfit import prepare_outfit_pool, compose_outfit, validate_and_finalize
from src.telemetry.timer import time_call


router = APIRouter()


@router.get("")
async def ping() -> dict[str, str]:
    return {"message": "outfit router ready"}


@router.post("")
async def outfit(body: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    query = (body or {}).get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    use_judge = bool(body.get("use_judge", settings.use_judge_default))
    debug = bool(body.get("debug", False))

    timings: Dict[str, float] = {}
    # Parse facets
    try:
        facets, ms = time_call(parse_query, query)
        timings["parse_ms"] = ms
    except (ValueError, ParserError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Retrieve and filter
    k_retrieve = settings.k_retrieve
    ret, ms = time_call(retrieve, query, k=k_retrieve)
    timings["retrieve_ms"] = ms
    candidates = ret.get("items", [])
    (kept, dropped), ms = time_call(apply_filters, candidates, facets, price_tolerance=settings.price_tolerance)
    timings["filter_ms"] = ms

    # Optional judge
    if use_judge:
        (judged, judge_trace), ms = time_call(judge_rerank, facets, kept, top_m=settings.judge_top_m)
        timings["judge_ms"] = ms
    else:
        judged, judge_trace = kept, {"enabled": False, "reranked": 0}

    # Prepare pool
    pool, ms = time_call(prepare_outfit_pool, facets, judged, settings.outfit_candidates_pool)
    timings["pool_ms"] = ms

    # Compose via LLM
    try:
        plan, ms = time_call(compose_outfit, facets, pool, settings)
        timings["compose_ms"] = ms
    except Exception as e:
        # Fallback: create minimal plan with required slots from pool
        from src.models.outfit import OutfitItem, OutfitPlan
        plan = OutfitPlan(slots={})
        for slot in settings.outfit_required_slots:
            chosen = next((p for p in pool if p.get("slot_guess") == slot), None)
            if chosen:
                plan.slots[slot] = [OutfitItem(id=str(chosen["id"]), slot=slot, title=chosen["title"], brand=chosen.get("brand"), price=chosen.get("price"))]
        if not plan.slots:
            raise HTTPException(status_code=422, detail=f"Composer failed: {e}")

    # Finalize
    (resp_out, comp_trace), ms = time_call(validate_and_finalize, plan, pool, facets, settings)
    timings["finalize_ms"] = ms

    trace = {
        "parsed": facets.model_dump(),
        "retrieval": {"k": k_retrieve, "hits": len(candidates)},
        "judge": judge_trace,
        "composer": {
            "model": settings.outfit_model,
            "pool": len(pool),
            "required_slots": settings.outfit_required_slots,
            **comp_trace,
        },
    }

    if debug:
        trace["timings"] = {k: round(v, 2) for k, v in timings.items()}
        resp_out["trace"] = trace
    # Ensure required fields exist (also when present as None)
    if ("total_price" not in resp_out) or (resp_out.get("total_price") is None):
        # best-effort compute
        total = 0.0
        slots = resp_out.get("slots", {}) or {}
        for items in slots.values():
            if items:
                p = items[0].get("price")
                if isinstance(p, (int, float)):
                    total += float(p)
        resp_out["total_price"] = round(total, 2)
    if ("under_budget" not in resp_out) or (resp_out.get("under_budget") is None):
        bud = facets.budget_usd.max if facets and facets.budget_usd else None
        tol = get_settings().outfit_price_tolerance
        if isinstance(bud, (int, float)):
            resp_out["under_budget"] = resp_out["total_price"] <= bud * (1.0 + tol)
        else:
            resp_out["under_budget"] = True
    return resp_out


