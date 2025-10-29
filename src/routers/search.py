from __future__ import annotations

from typing import Any, Dict, List
import math
from collections.abc import Mapping, Sequence
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from fastapi import APIRouter, HTTPException
import asyncio
from starlette.concurrency import run_in_threadpool

from src.core.config import get_settings
import uuid
from src.core.parser import ParserError, parse_query
from src.core.retriever import retrieve, get_store
from src.core.filters import apply_filters
from src.core.rerank import judge_rerank, should_enable_judge
from src.core.mmr import mmr_select
from src.telemetry.timer import time_call
from src.core import result_cache as result_cache


router = APIRouter()
def _to_jsonable(val: Any) -> Any:
    # Primitives
    if val is None or isinstance(val, (str, bool, int)):
        return val
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else val
    # Numpy scalars
    if np is not None and isinstance(val, getattr(np, "generic", ())):  # type: ignore[arg-type]
        return _to_jsonable(val.item())
    # Numpy arrays
    if np is not None and isinstance(val, getattr(np, "ndarray", ())):  # type: ignore[arg-type]
        try:
            return [_to_jsonable(x) for x in val.tolist()]
        except Exception:
            return [str(x) for x in val]
    # Mappings
    if isinstance(val, Mapping):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    # Sequences (but not strings)
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
        return [_to_jsonable(x) for x in val]
    # Fallback to string
    return val



@router.get("")
async def ping() -> dict[str, str]:
    return {"message": "search router ready"}


@router.post("")
async def search(body: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    query = (body or {}).get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    limit = body.get("limit", 12)
    try:
        limit = int(limit)
    except Exception:
        limit = 12
    limit = max(1, min(50, limit))
    page = body.get("page", 1)
    try:
        page = int(page)
    except Exception:
        page = 1
    page = max(1, page)
    offset = (page - 1) * limit
    user_flag = body.get("use_judge", None)
    result_id = (body or {}).get("result_id")
    debug = bool(body.get("debug", False))

    timings: Dict[str, float] = {}
    # Request enough candidates to cover all desired pages (use MMR_FINAL_K when configured)
    mmr_final_k_cfg = settings.mmr_final_k if settings.mmr_final_k > 0 else 0
    k_retrieve = max(offset + limit + 1, mmr_final_k_cfg)

    async def _timed_thread(fn, *args, **kwargs):
        start = asyncio.get_running_loop().time()
        res = await run_in_threadpool(fn, *args, **kwargs)
        end = asyncio.get_running_loop().time()
        return res, (end - start) * 1000.0

    # If client provided a result_id and cache is enabled, try serving from cache
    cached_items: List[Dict[str, Any]] | None = None
    if result_id and get_settings().result_cache_enable:
        cached_items = result_cache.get(str(result_id))  # may be None if expired/missing

    if cached_items is None:
        # Fire parse and retrieve concurrently
        parse_task = asyncio.create_task(_timed_thread(parse_query, query))
        retrieve_task = asyncio.create_task(_timed_thread(retrieve, query, k=k_retrieve))

        try:
            (facets, parse_ms), (ret, retrieve_ms) = await asyncio.gather(parse_task, retrieve_task)
            timings["parse_ms"] = parse_ms
            timings["retrieve_ms"] = retrieve_ms
        except Exception as e:
            # If parser failed, fall back to minimal facets instead of hard 422 so retrieval can proceed
            if isinstance(e, ParserError) or isinstance(getattr(e, "__cause__", None), ParserError):
                from src.models.schemas import QueryFacets
                facets = QueryFacets()  # minimal defaults
                timings["parse_ms"] = 0.0
                # still use retrieve result from the already awaited retrieve_task if available
                try:
                    ret_tuple = await retrieve_task
                    # ret_tuple is (res, ms)
                    ret = ret_tuple[0]
                    retrieve_ms = float(ret_tuple[1])
                except Exception:
                    raise HTTPException(status_code=422, detail=str(e))
                timings["retrieve_ms"] = retrieve_ms
            else:
                raise

        candidates: List[Dict[str, Any]] = (ret or {}).get("items", [])
        retrieval_trace = (ret or {}).get("trace", {})
        retrieval_hits = len(candidates)
    else:
        # Skip expensive steps; we'll only slice later
        facets = None  # type: ignore
        retrieval_trace = {"cached": True}
        retrieval_hits = len(cached_items or [])

    # Build ordered list (from scratch or cache)
    if cached_items is None:
        # Filters
        (kept, dropped_reasons), ms = time_call(apply_filters, candidates, facets, price_tolerance=settings.price_tolerance)
        timings["filter_ms"] = ms

        # Judge (optional heuristic)
        judged: List[Dict[str, Any]]
        # Decide judge enablement: explicit user flag wins; else auto policy; else default
        if isinstance(user_flag, bool):
            use_judge = user_flag
            auto_reason = "user_override"
        elif settings.judge_auto_enable:
            use_judge, auto_reason = should_enable_judge(facets, page, settings)
        else:
            use_judge = settings.use_judge_default
            auto_reason = "default"

        if use_judge:
            # Adaptive top_m based on simple signal count
            signals = 0
            if facets.budget_usd and facets.budget_usd.max is not None:
                signals += 1
            if facets.colors:
                signals += 1
            if facets.materials:
                signals += 1
            if facets.categories_include:
                signals += 1
            top_m_dynamic = settings.judge_top_m if signals >= 2 else min(settings.judge_top_m, 12)
            # Provide hybrid context for gate (if available)
            hybrid = (retrieval_trace or {}).get("hybrid", {})
            dense_ids = hybrid.get("dense_ids") if isinstance(hybrid, dict) else None
            lex_ids = hybrid.get("lex_ids") if isinstance(hybrid, dict) else None
            (judged, judge_trace), ms = time_call(
                judge_rerank,
                facets,
                kept,
                top_m=top_m_dynamic,
                dense_ids=dense_ids,
                lex_ids=lex_ids,
                normalized_query=(query or "").strip().lower(),
            )
            timings["judge_ms"] = ms
        else:
            judged = kept
            judge_trace = {"enabled": False, "reranked": 0}
            timings["judge_ms"] = 0.0

        # Decide target size for the ordered list (covers page windows)
        final_k = max(k_retrieve, settings.mmr_final_k if settings.mmr_final_k > 0 else 0)

        # If too few after filters, backfill from unfiltered candidates to reach final_k
        if len(judged) < final_k:
            have_ids = {str(it.get("id")) for it in judged}
            for it in candidates:
                pid = str(it.get("id"))
                if pid and pid not in have_ids:
                    judged.append(it)
                    have_ids.add(pid)
                if len(judged) >= final_k:
                    break

        # Diversify (MMR) — cap candidates for speed
        judged_for_mmr = sorted(judged, key=lambda it: float(it.get("score") or it.get("raw_score") or 0.0), reverse=True)[: settings.mmr_candidates_max]
        (diversified, mmr_trace), ms = time_call(
            mmr_select,
            judged_for_mmr,
            lambda_mult=settings.mmr_lambda,
            final_k=final_k,
            mode=settings.mmr_similarity_mode,
            title_threshold=settings.mmr_title_sim_threshold,
            dedup_keys=settings.dedup_keys,
            variant_keys=settings.variant_keys,
        )
        timings["mmr_ms"] = ms

        # Store ordered list in result cache (page 1 or when no result_id provided)
        new_result_id = None
        if get_settings().result_cache_enable and (not result_id) and page == 1:
            new_result_id = result_cache.put(diversified)
        # Fallback: ensure we always return a non-empty result_id so clients can pass it back
        result_id = new_result_id or result_id or uuid.uuid4().hex
        # If client supplied a result_id but cache miss, write into cache under that id for stable pagination
        if get_settings().result_cache_enable and (new_result_id is None) and (body.get("result_id")):
            try:
                result_cache.put_with_id(str(body.get("result_id")), diversified)
            except Exception:
                pass
    else:
        # Cached path: restore diversified list and skip filters/judge/mmr
        diversified = cached_items  # type: ignore
        mmr_trace = {"cached": True}
        judge_trace = {"enabled": False, "cached": True}
        dropped_reasons = {}
        kept = diversified
        auto_reason = "cache"

    # Rationale fallback
    if diversified:
        if 'facets' in locals() and facets is not None:
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

    # Page slice
    page_slice = diversified[offset:offset+limit]
    has_more = len(diversified) > (offset + limit)

    items = []
    for it in page_slice:
        # Start with all retriever fields so rich metadata is preserved to the client
        merged: Dict[str, Any] = dict(it)
        # Normalize core fields and ensure score is numeric
        merged["id"] = it.get("id")
        # Only include title if present (frontend expects string when provided)
        if isinstance(it.get("title"), str):
            merged["title"] = it.get("title")
        else:
            merged.pop("title", None)
        merged["brand"] = it.get("brand")
        merged["price"] = it.get("price")
        merged["image_url"] = it.get("image_url")
        merged["product_url"] = it.get("product_url")
        merged["score"] = float(it.get("score") or it.get("raw_score") or 0.0)
        merged["why"] = it.get("why")
        items.append(_to_jsonable(merged))

    # Summary and trace
    if 'facets' in locals() and facets is not None:
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
    else:
        summary = f"{len(items)} recs."

    trace = {
        "parsed": (facets.model_dump() if ('facets' in locals() and facets is not None) else {"skipped": True}),
        "retrieval": {
            "k": k_retrieve,
            "hits": retrieval_hits,
            "dim": retrieval_trace.get("dim"),
            "ntotal": retrieval_trace.get("ntotal"),
            "hybrid": retrieval_trace.get("hybrid"),
        },
        "filters": {
            "kept": len(kept) if 'kept' in locals() else len(diversified),
            "dropped": (retrieval_hits - len(kept)) if 'kept' in locals() else 0,
            "reasons": dropped_reasons if 'dropped_reasons' in locals() else {},
        },
        "judge": {**judge_trace, "auto": settings.judge_auto_enable and not isinstance(user_flag, bool), "reason": auto_reason if 'auto_reason' in locals() else "cache"},
        "mmr": {**mmr_trace},
    }
    if isinstance(judge_trace, dict) and "gate" in judge_trace:
        trace["judge_gate"] = judge_trace.get("gate")
    trace["timings"] = {k: round(v, 2) for k, v in timings.items()}

    resp: Dict[str, Any] = {"items": items, "summary": summary, "page": page, "limit": limit, "has_more": has_more}
    if result_id:
        resp["result_id"] = result_id
    if debug:
        resp["trace"] = trace
    return resp


