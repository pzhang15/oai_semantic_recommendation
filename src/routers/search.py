from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.core.config import get_settings
from src.core.parser import ParserError, parse_query
from src.core.retriever import retrieve, get_store
from src.core.filters import apply_filters
from src.core.rerank import judge_rerank
from src.core.mmr import mmr_select
from src.telemetry.timer import time_call


router = APIRouter()


@router.get("")
async def ping() -> dict[str, str]:
    return {"message": "search router ready"}


@router.post("")
async def search(body: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    query = (body or {}).get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    limit = body.get("limit", settings.top_k_default)
    try:
        limit = int(limit)
    except Exception:
        limit = settings.top_k_default
    limit = max(1, min(50, limit))
    use_judge = body.get("use_judge", settings.use_judge_default)
    debug = bool(body.get("debug", False))

    timings: Dict[str, float] = {}
    # Parse
    try:
        facets, ms = time_call(parse_query, query)
        timings["parse_ms"] = ms
    except (ValueError, ParserError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Retrieve
    k_retrieve = settings.k_retrieve
    ret, ms = time_call(retrieve, query, k=k_retrieve)
    timings["retrieve_ms"] = ms
    candidates: List[Dict[str, Any]] = ret.get("items", [])
    retrieval_trace = ret.get("trace", {})
    retrieval_hits = len(candidates)

    # Filters
    (kept, dropped_reasons), ms = time_call(apply_filters, candidates, facets, price_tolerance=settings.price_tolerance)
    timings["filter_ms"] = ms

    # Judge (optional heuristic)
    judged: List[Dict[str, Any]]
    if use_judge:
        (judged, judge_trace), ms = time_call(judge_rerank, facets, kept, top_m=settings.judge_top_m)
        timings["judge_ms"] = ms
    else:
        judged = kept
        judge_trace = {"enabled": False, "reranked": 0}

    # If too few after filters, backfill from unfiltered candidates to reach limit
    if len(judged) < limit:
        have_ids = {str(it.get("id")) for it in judged}
        for it in candidates:
            pid = str(it.get("id"))
            if pid and pid not in have_ids:
                judged.append(it)
                have_ids.add(pid)
            if len(judged) >= limit:
                break

    # Diversify (MMR)
    final_k = settings.mmr_final_k if settings.mmr_final_k > 0 else limit
    (diversified, mmr_trace), ms = time_call(
        mmr_select,
        judged,
        lambda_mult=settings.mmr_lambda,
        final_k=final_k,
        mode=settings.mmr_similarity_mode,
        title_threshold=settings.mmr_title_sim_threshold,
        dedup_keys=settings.dedup_keys,
        variant_keys=settings.variant_keys,
    )
    timings["mmr_ms"] = ms

    # Rationale fallback
    for it in diversified:
        if not it.get("why"):
            hints: List[str] = []
            if facets.budget_usd.max is not None and isinstance(it.get("price"), (int, float)) and it.get("price") <= facets.budget_usd.max * (1.0 + settings.price_tolerance):
                hints.append("within budget")
            for m in facets.materials or []:
                if m in (it.get("title") or "").lower():
                    hints.append(m)
            for c in facets.colors or []:
                if c in (it.get("title") or "").lower():
                    hints.append(c)
            if hints:
                it["why"] = ", ".join(sorted(set(hints)))

    items = []
    for it in diversified:
        items.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "brand": it.get("brand"),
            "price": it.get("price"),
            "image_url": it.get("image_url"),
            "product_url": it.get("product_url"),
            "score": float(it.get("score") or it.get("raw_score") or 0.0),
            "why": it.get("why"),
        })

    # Summary and trace
    occ = facets.occasion or ""
    season = facets.season or ""
    bud = facets.budget_usd.max
    summary_bits = [f"{len(items)} recs"]
    if season:
        summary_bits.append(season)
    if occ:
        summary_bits.append(occ)
    if isinstance(bud, (int, float)):
        summary_bits.append(f"under ${int(bud)}")
    summary = " ".join(summary_bits) + "."

    trace = {
        "parsed": facets.model_dump(),
        "retrieval": {
            "k": k_retrieve,
            "hits": retrieval_hits,
            "dim": retrieval_trace.get("dim"),
            "ntotal": retrieval_trace.get("ntotal"),
        },
        "filters": {
            "kept": len(kept),
            "dropped": retrieval_hits - len(kept),
            "reasons": dropped_reasons,
        },
        "judge": judge_trace,
        "mmr": {**mmr_trace},
    }
    trace["timings"] = {k: round(v, 2) for k, v in timings.items()}

    resp: Dict[str, Any] = {"items": items, "summary": summary}
    if debug:
        resp["trace"] = trace
    return resp


