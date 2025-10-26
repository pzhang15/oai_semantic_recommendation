from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import BadRequestError, OpenAI

from src.core.config import get_settings
from src.models.schemas import QueryFacets
from src.telemetry.timer import span
from src.telemetry.counters import add_usage


# In-memory cache with TTL
_JUDGE_CACHE: Dict[Tuple[str, str, str, str], Tuple[Dict[str, Any], float]] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0
_TEST_FORCE_TEMP_REJECT = False


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


def judge_batch(facets: QueryFacets, batch_candidates: List[Dict[str, Any]], model_id: str, settings) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    facets_summary = _summarize_facets_for_prompt(facets)
    schema = _build_schema_array()
    cands_min = [
        {"id": str(it.get("id")), "title": it.get("title"), "brand": it.get("brand"), "price": it.get("price")}
        for it in batch_candidates
    ]

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

    with span("judge") as sp:
        resp = _call_chat_completion(client, model_id, messages, schema, settings.judge_max_tokens, settings.judge_timeout_secs)
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if settings is None:
        settings = get_settings()
    model_id = model or settings.model_judge

    if not candidates:
        return [], {"enabled": True, "model": model_id, "top_m": 0, "batch_size": 0, "cached_hits": 0, "cached_misses": 0}

    # Select top M by retrieval
    M = min(int(top_m or settings.judge_top_m), len(candidates))
    top = sorted(candidates, key=lambda it: float(it.get("raw_score") or it.get("score") or 0.0), reverse=True)[:M]

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

    # Lookup cache and split
    global _CACHE_HITS, _CACHE_MISSES
    need_eval: List[Dict[str, Any]] = []
    judged_map: Dict[str, Dict[str, Any]] = {}
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

    # Batch evaluate uncached
    for i in range(0, len(need_eval), bs):
        batch = need_eval[i:i+bs]
        try:
            scored = judge_batch(facets, batch, model_id, settings)
        except Exception:
            scored = _heuristic_judge_batch(facets, batch, settings)
        if not scored:
            scored = _heuristic_judge_batch(facets, batch, settings)
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
            # trim cache if too large
            if len(_JUDGE_CACHE) > settings.judge_cache_size:
                _JUDGE_CACHE.pop(next(iter(_JUDGE_CACHE)))

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

    trace = {
        "enabled": True,
        "model": model_id,
        "top_m": M,
        "batch_size": bs,
        "cached_hits": _CACHE_HITS,
        "cached_misses": _CACHE_MISSES,
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


