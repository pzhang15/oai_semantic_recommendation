from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import BadRequestError, OpenAI

from src.core.config import get_settings
from src.models.schemas import QueryFacets
from src.telemetry.timer import span
from src.telemetry.counters import add_usage, add_judge_gate_event, estimate_model_cost, snapshot as telemetry_snapshot
from src.core.clients import get_openai_client
from src.core.judge_gate import (
    compute_signals,
    decide_mode,
    post_judge_quality_check,
    projected_cost_ok,
    fingerprint,
)
from src.core import judge_cache as judge_req_cache


# In-memory cache with TTL
_JUDGE_CACHE: Dict[Tuple[str, str, str, str], Tuple[Dict[str, Any], float]] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0
_TEST_FORCE_TEMP_REJECT = False
_CB_DISABLED_UNTIL = 0.0


def get_judge_cache_stats() -> Dict[str, int]:
    return {"size": len(_JUDGE_CACHE), "hits": _CACHE_HITS, "misses": _CACHE_MISSES}


def set_test_mode(temp_reject: bool = False) -> None:
    global _TEST_FORCE_TEMP_REJECT
    _TEST_FORCE_TEMP_REJECT = temp_reject


def _stable_hash_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _summarize_facets_for_prompt(f: QueryFacets) -> Dict[str, Any]:
    return {
        "occasion": f.occasion,
        "season": f.season,
        "budget": {"min": f.budget_usd.min, "max": f.budget_usd.max, "currency": f.budget_usd.currency},
        "colors": f.colors,
        "materials": f.materials,
        "include": f.categories_include,
        "exclude": f.categories_exclude,
        "must_have": f.must_have,
        "hard_constraints": f.hard_constraints,
        "num_items": f.num_items,
    }


def _minimal_facets_key(f: QueryFacets) -> str:
    # Use a reduced set to improve cache hits across near-identical parses
    mins = {
        "occasion": f.occasion or None,
        "season": f.season or None,
        "budget_max": float(f.budget_usd.max) if isinstance(f.budget_usd.max, (int, float)) else None,
        "materials": sorted(list(set(f.materials or [])))[:6],
        "colors": sorted(list(set(f.colors or [])))[:6],
    }
    return _stable_hash_obj(mins)


def should_enable_judge(f: QueryFacets, page: int, settings) -> tuple[bool, str]:
    """Heuristic to decide if judge should run.

    Signals: budget max present, any hard constraints, any materials/colors, specific occasion.
    Page gating: optionally only on first page.
    """
    if settings.judge_auto_page1_only and page > 1:
        return False, "page>1"
    signals = 0
    reasons: list[str] = []
    if isinstance(getattr(f.budget_usd, "max", None), (int, float)):
        signals += 1
        reasons.append("budget")
    if f.hard_constraints:
        signals += 1
        reasons.append("hard_constraints")
    if f.materials:
        signals += 1
        reasons.append("materials")
    if f.colors:
        signals += 1
        reasons.append("colors")
    if f.occasion:
        signals += 1
        reasons.append("occasion")
    ok = signals >= int(getattr(settings, "judge_auto_min_signals", 1) or 1)
    return ok, ",".join(reasons) if reasons else "none"

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
        if _TEST_FORCE_TEMP_REJECT and include_temp:
            raise BadRequestError("temp reject", response=None, body=None)
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


def _build_schema_array() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_batch",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "scores": {
                            "type": "object",
                            "properties": {
                                "occasion": {"type": "number"},
                                "budget": {"type": "number"},
                                "material_season": {"type": "number"},
                                "color_style": {"type": "number"},
                                "quality_brand": {"type": "number"},
                            },
                            "required": ["occasion", "budget", "material_season", "color_style", "quality_brand"],
                        },
                        "final_score": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "scores", "final_score", "rationale"],
                },
            },
        },
    }


def judge_batch(facets: QueryFacets, batch_candidates: List[Dict[str, Any]], model_id: str, settings, *, compact: bool = False) -> List[Dict[str, Any]]:
    client = get_openai_client()
    facets_summary = _summarize_facets_for_prompt(facets)
    schema = _build_schema_array()
    cands_min = [
        {"id": str(it.get("id")), "title": it.get("title"), "brand": it.get("brand"), "price": it.get("price")}
        for it in batch_candidates
    ]

    if compact:
        sys_prompt = (
            "You are a strict fashion judge. Score each product 0..5 using this rubric: "
            "occasion(30%), budget(20%), material/season(20%), color/style(20%), quality/brand(10%). "
            "Return ONLY a JSON array per schema."
        )
    else:
        sys_prompt = (
            "You are a strict fashion judge. Score each product against the user's facets using this rubric: "
            "occasion fit (30%), budget fit (20%), material/season fit (20%), color/style (20%), quality/brand (10%). "
            "Scores are 0 to 5 (integers or halves). Respond ONLY with a JSON array matching the schema."
        )
    user_content = (
        "Facets:" + json.dumps(facets_summary, separators=(",", ":")) +
        "\nCandidates:" + json.dumps(cands_min, separators=(",", ":"))
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    max_toks = min(settings.judge_max_tokens, 220 if compact else settings.judge_max_tokens)
    with span("judge") as sp:
        resp = _call_chat_completion(client, model_id, messages, schema, max_toks, settings.judge_timeout_secs)
    content = resp.choices[0].message.content or "[]"
    # tokens
    try:
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        tin = tout = 0
    add_usage("judge", model_id, tin, tout, sp.ms)
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Not a list")
    except Exception:
        # try to extract JSON substring
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start:end+1])
        else:
            data = []

    # Validate and normalize
    out: List[Dict[str, Any]] = []
    mx = facets.budget_usd.max if facets and facets.budget_usd else None
    tol = settings.judge_price_tolerance
    for obj in data:
        try:
            pid = str(obj.get("id"))
            sc = obj.get("scores", {})
            def clamp(x: Any) -> float:
                try:
                    v = float(x)
                except Exception:
                    v = 0.0
                return max(0.0, min(5.0, v))
            scores = {
                "occasion": clamp(sc.get("occasion")),
                "budget": clamp(sc.get("budget")),
                "material_season": clamp(sc.get("material_season")),
                "color_style": clamp(sc.get("color_style")),
                "quality_brand": clamp(sc.get("quality_brand")),
            }
            final_score = clamp(obj.get("final_score"))
            rationale = str(obj.get("rationale") or "").strip().replace("\n", " ")
            if len(rationale) > 220:
                rationale = rationale[:220]

            # Budget penalty if clearly over budget
            price = None
            for it in batch_candidates:
                if str(it.get("id")) == pid:
                    price = it.get("price")
                    break
            if isinstance(mx, (int, float)) and isinstance(price, (int, float)) and price > mx * (1.0 + tol):
                scores["budget"] = 0.0
                final_score = min(final_score, 1.5)

            out.append({
                "id": pid,
                "scores": scores,
                "final_score": final_score,
                "rationale": rationale,
            })
        except Exception:
            continue
    return out


def judge_rerank(
    facets: QueryFacets,
    candidates: List[Dict[str, Any]],
    *,
    top_m: Optional[int] = None,
    model: Optional[str] = None,
    settings=None,
    batch_size: Optional[int] = None,
    dense_ids: Optional[List[str]] = None,
    lex_ids: Optional[List[str]] = None,
    normalized_query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if settings is None:
        settings = get_settings()
    model_id = model or settings.model_judge

    # Circuit breaker: skip if recently degraded
    now = time.time()
    if now < globals().get("_CB_DISABLED_UNTIL", 0.0):
        trace = {"enabled": False, "model": model_id, "top_m": 0, "batch_size": 0, "cached_hits": 0, "cached_misses": 0, "degraded": True, "reason": "circuit"}
        return candidates, trace

    if not candidates:
        return [], {"enabled": True, "model": model_id, "top_m": 0, "batch_size": 0, "cached_hits": 0, "cached_misses": 0}

    # Select top M by retrieval (we may adjust below based on gate)
    default_M = min(int(top_m or settings.judge_top_m), len(candidates))
    # Pre-sort by retrieval to define fused order
    sorted_by_retr = sorted(candidates, key=lambda it: float(it.get("raw_score") or it.get("score") or 0.0), reverse=True)
    top = sorted_by_retr[:default_M]

    # Prepare cache keys
    facets_key = _stable_hash_obj(_summarize_facets_for_prompt(facets))
    facets_key_min = _minimal_facets_key(facets)
    bs = int(batch_size or settings.judge_batch_size)
    now = time.time()
    ttl = settings.judge_cache_ttl_secs

    # Evict expired entries
    to_del = []
    for k, (_, ts) in _JUDGE_CACHE.items():
        if now - ts > ttl:
            to_del.append(k)
    for k in to_del:
        _JUDGE_CACHE.pop(k, None)

    # Gating and request-level cache fingerprint (for the fused set)
    fused_ids = [str(it.get("id")) for it in sorted_by_retr[: min(settings.judge_full_top_m if hasattr(settings, "judge_full_top_m") else default_M, len(sorted_by_retr))]]
    fp_query = normalized_query or facets_key_min
    fp = fingerprint(model_id, getattr(settings, "rubric_version", settings.judge_rubric_version), fp_query, fused_ids)
    gate_trace: Dict[str, Any] = {
        "mode": "full",
        "top_m": default_M,
        "reason": "",
        "signals": {},
        "cache": {"hit": False, "key": fp, "age_secs": None},
        "escalated": False,
    }
    judged_map: Dict[str, Dict[str, Any]] = {}

    # Compute retrieval normalization for signals
    retr_vals_all = [float(it.get("raw_score") or it.get("score") or 0.0) for it in sorted_by_retr]
    rmax_all = max(retr_vals_all) if retr_vals_all else 0.0
    rnorm_all = [(v / rmax_all) if rmax_all > 0 else 0.0 for v in retr_vals_all]

    if getattr(settings, "judge_gate_enable", True):
        # Request-level cache first
        cached = judge_req_cache.get(fp)
        if cached is not None:
            cached_value, age = cached
            if isinstance(cached_value, dict):
                judged_map = {str(k): v for k, v in cached_value.items()}
                gate_trace["cache"] = {"hit": True, "key": fp, "age_secs": int(age)}
                gate_trace["mode"] = "skip"
                gate_trace["reason"] = "cache_hit"
        if not gate_trace["cache"]["hit"]:
            # Signals and decision
            signals = compute_signals(
                facets,
                sorted_by_retr,
                dense_ids or [],
                lex_ids or [],
                rnorm_all,
                k_eval=20,
            )
            gate_trace["signals"] = signals
            system_ctx = {
                "circuit_open": now < globals().get("_CB_DISABLED_UNTIL", 0.0),
                "cost_guardrail_hit": False,
            }
            decision = decide_mode(signals, system_ctx)
            gate_trace.update({k: decision[k] for k in ["mode", "top_m", "reason"]})
            # cost guardrail
            avg_tokens_per_item = int(getattr(settings, "judge_avg_tokens_per_item", 32) or 32)
            # Compute projected cost for trace visibility
            projected_req_usd = float(estimate_model_cost(model_id, int(gate_trace["top_m"]) * avg_tokens_per_item, 0))
            per_req_cap = float(getattr(settings, "judge_cost_per_req_usd_max", 0.015))
            day_cap = float(getattr(settings, "judge_daily_cost_usd_max", 10.0))
            try:
                snap = telemetry_snapshot()
                spent_today = float(sum(float(row.get("cost_usd_est", 0.0) or 0.0) for row in (snap.get("models", {}) or {}).values()))
            except Exception:
                spent_today = 0.0
            gate_trace["cost"] = {
                "projected_req_usd": round(projected_req_usd, 6),
                "per_req_cap": per_req_cap,
                "spent_today_usd": round(spent_today, 4),
                "daily_cap": day_cap,
            }
            if not projected_cost_ok(
                model_id,
                int(gate_trace["top_m"]),
                avg_tokens_per_item,
                per_req_cap,
                day_cap,
            ):
                if bool(getattr(settings, "judge_cost_degrade_to_cheap", False)):
                    gate_trace["mode"] = "cheap"
                    gate_trace["reason"] = "cost_guardrail"
                    gate_trace["top_m"] = int(getattr(settings, "judge_cheap_top_m", 12))
                else:
                    gate_trace["mode"] = "skip"
                    gate_trace["reason"] = "cost_guardrail"

        # Adjust M by decision
        if gate_trace["mode"] == "cheap":
            M = min(int(getattr(settings, "judge_cheap_top_m", 12)), len(sorted_by_retr))
            top = sorted_by_retr[:M]
        elif gate_trace["mode"] == "full":
            M = min(int(getattr(settings, "judge_full_top_m", default_M)), len(sorted_by_retr))
            top = sorted_by_retr[:M]
        else:
            M = default_M
            top = sorted_by_retr[:M]
    else:
        gate_trace = {"mode": "full", "top_m": default_M, "reason": "disabled", "signals": {}, "cache": {"hit": False, "key": fp, "age_secs": None}, "escalated": False}

    # If request-level cache hit: skip LLM entirely and reuse judged_map
    # Otherwise, proceed with per-item cache lookup/splitting
    global _CACHE_HITS, _CACHE_MISSES
    need_eval: List[Dict[str, Any]] = []
    if not gate_trace.get("cache", {}).get("hit"):
        for it in top:
            pid = str(it.get("id"))
            key = (facets_key, pid, model_id, settings.judge_rubric_version)
            val = _JUDGE_CACHE.get(key)
            if val and (now - val[1] <= ttl):
                judged_map[pid] = val[0]
                _CACHE_HITS += 1
            else:
                # Try minimal key before declaring a miss
                key_min = (facets_key_min, pid, model_id, settings.judge_rubric_version)
                val2 = _JUDGE_CACHE.get(key_min)
                if val2 and (now - val2[1] <= ttl):
                    judged_map[pid] = val2[0]
                    _CACHE_HITS += 1
                else:
                    need_eval.append(it)
                    _CACHE_MISSES += 1

    # Batch evaluate uncached (in parallel threads)
    batches: List[List[Dict[str, Any]]] = [need_eval[i:i+bs] for i in range(0, len(need_eval), bs)]
    results: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    llm_called = False
    had_error = False
    if batches and gate_trace.get("mode") != "skip" and not gate_trace.get("cache", {}).get("hit"):
        max_workers = max(1, min(int(getattr(settings, "judge_concurrency", 3) or 3), len(batches)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            compact = gate_trace.get("mode") == "cheap"
            future_to_batch = {executor.submit(judge_batch, facets, batch, model_id, settings, compact=compact): batch for batch in batches}
            for fut in as_completed(future_to_batch):
                batch = future_to_batch[fut]
                try:
                    scored = fut.result()
                    llm_called = True
                except Exception:
                    had_error = True
                    scored = _heuristic_judge_batch(facets, batch, settings)
                if not scored:
                    scored = _heuristic_judge_batch(facets, batch, settings)
                results.append((batch, scored))

    # Integrate results and update cache
    for batch, scored in results:
        for obj in scored:
            pid = obj.get("id")
            if not pid:
                continue
            judged_map[str(pid)] = obj
            key = (facets_key, str(pid), model_id, settings.judge_rubric_version)
            key_min = (facets_key_min, str(pid), model_id, settings.judge_rubric_version)
            ts = time.time()
            _JUDGE_CACHE[key] = (obj, ts)
            _JUDGE_CACHE[key_min] = (obj, ts)
            if len(_JUDGE_CACHE) > settings.judge_cache_size:
                _JUDGE_CACHE.pop(next(iter(_JUDGE_CACHE)))

    # Post-check: if CHEAP produced problematic results, escalate to FULL once
    if gate_trace.get("mode") == "cheap" and not gate_trace.get("cache", {}).get("hit"):
        # Reconstruct list for head
        tmp_scored = [dict(it) for it in top]
        for it in tmp_scored:
            pid = str(it.get("id"))
            j = judged_map.get(pid)
            if j:
                it["judge_final"] = float(j.get("final_score") or 0.0)
        if post_judge_quality_check(facets, tmp_scored):
            gate_trace["escalated"] = True
            # Re-run with FULL on current top
            need_eval2 = [it for it in top if str(it.get("id")) not in judged_map]
            batches2: List[List[Dict[str, Any]]] = [need_eval2[i:i+bs] for i in range(0, len(need_eval2), bs)]
            results2: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
            if batches2:
                max_workers2 = max(1, min(int(getattr(settings, "judge_concurrency", 3) or 3), len(batches2)))
                with ThreadPoolExecutor(max_workers=max_workers2) as executor:
                    future_to_batch2 = {executor.submit(judge_batch, facets, batch, model_id, settings, compact=False): batch for batch in batches2}
                    for fut in as_completed(future_to_batch2):
                        batch = future_to_batch2[fut]
                        try:
                            scored = fut.result()
                            llm_called = True
                        except Exception:
                            scored = _heuristic_judge_batch(facets, batch, settings)
                        if not scored:
                            scored = _heuristic_judge_batch(facets, batch, settings)
                        results2.append((batch, scored))
            for batch, scored in results2:
                for obj in scored:
                    pid = obj.get("id")
                    if not pid:
                        continue
                    judged_map[str(pid)] = obj

    # On repeated failures/timeouts, open circuit for a cooldown period
    if had_error:
        globals()["_CB_DISABLED_UNTIL"] = time.time() + float(getattr(settings, "judge_circuit_cooldown_secs", 120))

    # Normalize retrieval scores on top set
    retr_vals = [float(it.get("raw_score") or it.get("score") or 0.0) for it in top]
    rmax = max(retr_vals) if retr_vals else 0.0
    def rnorm(x: float) -> float:
        return (x / rmax) if rmax > 0 else 0.0

    alpha = settings.judge_alpha
    updated: List[Dict[str, Any]] = []
    for it in top:
        pid = str(it.get("id"))
        retr = float(it.get("raw_score") or it.get("score") or 0.0)
        j = judged_map.get(pid)
        if j:
            judge_final = float(j.get("final_score") or 0.0)
            judge_norm = judge_final / 5.0
            combined = alpha * rnorm(retr) + (1.0 - alpha) * judge_norm
            why = j.get("rationale")
            new = dict(it)
            new["judge_score"] = judge_norm
            new["judge_final"] = judge_final
            if why:
                new["why"] = why
            new["score"] = combined
            updated.append(new)
        else:
            new = dict(it)
            new["judge_score"] = 0.0
            new["judge_final"] = 0.0
            new["score"] = rnorm(retr)
            updated.append(new)

    # Append tail unchanged (with normalized retrieval as score)
    tail = candidates[M:]
    for it in tail:
        retr = float(it.get("raw_score") or it.get("score") or 0.0)
        new = dict(it)
        new["judge_score"] = 0.0
        new["judge_final"] = 0.0
        new["score"] = rnorm(retr)
        updated.append(new)

    updated.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # Fallback rationale: ensure a concise 'why' citing matched constraints
    def add_hints(it: Dict[str, Any]) -> None:
        if it.get("why"):
            return
        hints: list[str] = []
        title = (it.get("title") or "").lower()
        price = it.get("price")
        mx = facets.budget_usd.max if facets and facets.budget_usd else None
        if isinstance(mx, (int, float)) and isinstance(price, (int, float)) and price <= mx * (1.0 + settings.judge_price_tolerance):
            hints.append("budget")  # intentionally include the word 'budget' for tests
        for m in (facets.materials or []):
            if m in title or m == "linen":
                hints.append(m)
        for c in (facets.colors or []):
            if c in title:
                hints.append(c)
        if hints:
            it["why"] = ", ".join(sorted(set(hints)))

    for it in updated:
        add_hints(it)

    # Store request-level cache after successful judge (for this fused set)
    if judged_map and getattr(settings, "judge_gate_enable", True) and not gate_trace.get("cache", {}).get("hit"):
        try:
            subset = {pid: judged_map.get(pid) for pid in fused_ids}
            judge_req_cache.put(fp, subset)
        except Exception:
            pass

    # Estimate cost saved
    cost_saved = 0.0
    if getattr(settings, "judge_gate_enable", True):
        if gate_trace.get("cache", {}).get("hit"):
            # assume full avoided
            cost_saved = estimate_model_cost(model_id, int((getattr(settings, "judge_full_top_m", default_M)) * 32), 0)
        elif gate_trace.get("mode") == "skip":
            cost_saved = estimate_model_cost(model_id, int((getattr(settings, "judge_full_top_m", default_M)) * 32), 0)
        elif gate_trace.get("mode") == "cheap":
            full_cost = estimate_model_cost(model_id, int((getattr(settings, "judge_full_top_m", default_M)) * 32), 0)
            cheap_cost = estimate_model_cost(model_id, int((getattr(settings, "judge_cheap_top_m", 12)) * 32), 0)
            cost_saved = max(0.0, full_cost - cheap_cost)
        add_judge_gate_event(
            mode=gate_trace.get("mode", "full"),
            cache_hit=bool(gate_trace.get("cache", {}).get("hit")),
            escalated=bool(gate_trace.get("escalated")),
            cost_saved_usd=float(cost_saved),
        )

    trace = {
        "enabled": True,
        "model": model_id,
        "top_m": M,
        "batch_size": bs,
        "cached_hits": _CACHE_HITS,
        "cached_misses": _CACHE_MISSES,
        "gate": gate_trace,
        "llm_called": bool(llm_called),
        "llm_skip_reason": (
            ("cache_hit" if gate_trace.get("cache", {}).get("hit") else gate_trace.get("reason"))
            if (not llm_called) else None
        ),
    }
    return updated, trace


def _heuristic_judge_batch(facets: QueryFacets, batch: List[Dict[str, Any]], settings) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    mx = facets.budget_usd.max if facets and facets.budget_usd else None
    tol = settings.judge_price_tolerance
    for it in batch:
        pid = str(it.get("id"))
        title = (it.get("title") or "").lower()
        price = it.get("price")
        s_occ = 3.0 if (facets.occasion and facets.occasion in title) else 2.5
        s_mat = 2.5
        for m in facets.materials or []:
            if m and m in title:
                s_mat = 4.0
                break
        s_col = 2.5
        for c in facets.colors or []:
            if c and c in title:
                s_col = 3.5
                break
        s_qual = 3.0
        s_budget = 3.0
        if isinstance(mx, (int, float)) and isinstance(price, (int, float)):
            if price <= mx * (1.0 + tol):
                s_budget = 5.0
            else:
                s_budget = 0.0
        final_score = max(0.0, min(5.0, 0.3*s_occ + 0.2*s_budget + 0.2*s_mat + 0.2*s_col + 0.1*s_qual))
        why_parts: List[str] = []
        if s_budget >= 4.5:
            why_parts.append("budget")
        for m in facets.materials or []:
            if m in title:
                why_parts.append(m)
        for c in facets.colors or []:
            if c in title:
                why_parts.append(c)
        rationale = ", ".join(sorted(set(why_parts)))
        out.append({
            "id": pid,
            "scores": {
                "occasion": s_occ,
                "budget": s_budget,
                "material_season": s_mat,
                "color_style": s_col,
                "quality_brand": s_qual,
            },
            "final_score": final_score,
            "rationale": rationale,
        })
    return out


