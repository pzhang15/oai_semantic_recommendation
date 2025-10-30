from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Sequence, Tuple

from src.core.config import get_settings
from src.telemetry import counters as telemetry_counters


def _softmax(vals: Sequence[float], tau: float) -> List[float]:
    if not vals:
        return []
    # Stabilize by subtracting max/tau
    m = max(vals)
    exps = [math.exp((v - m) / max(tau, 1e-6)) for v in vals]
    s = sum(exps)
    return [x / s for x in exps] if s > 0 else [0.0 for _ in exps]


def _entropy(probs: Sequence[float]) -> float:
    e = 0.0
    for p in probs:
        if p > 0:
            e -= p * math.log(p + 1e-12)
    return float(e)


def _norm_title_stem(title: str | None) -> str:
    if not isinstance(title, str):
        return ""
    t = title.lower()
    keep: list[str] = []
    for w in t.replace("/", " ").replace("-", " ").split():
        w2 = "".join(ch for ch in w if ch.isalnum())
        if len(w2) >= 3 and w2 not in {"with", "and", "for", "the", "men", "women", "kids"}:
            keep.append(w2)
    return " ".join(keep[:12])


def compute_signals(
    facets: Any,
    fused_candidates: List[Dict[str, Any]],
    dense_ids: Sequence[str] | None,
    lex_ids: Sequence[str] | None,
    retrieval_scores: Sequence[float],
    *,
    k_eval: int = 20,
) -> Dict[str, float]:
    s = get_settings()
    k = min(int(k_eval), len(fused_candidates)) if fused_candidates else 0

    # Specificity: categories/colors/materials/occasion/budget
    specificity = 0
    try:
        if getattr(facets, "categories_include", None):
            specificity += 1
        if getattr(facets, "colors", None):
            specificity += 1
        if getattr(facets, "materials", None):
            specificity += 1
        if getattr(facets, "occasion", None):
            specificity += 1
        bud_max = getattr(getattr(facets, "budget_usd", None), "max", None)
        if isinstance(bud_max, (int, float)):
            specificity += 1
    except Exception:
        pass

    # Margin and entropy on retrieval scores
    margin = 0.0
    entropy = 0.0
    if retrieval_scores:
        K = min(k if k > 0 else len(retrieval_scores), len(retrieval_scores))
        if K >= 1:
            margin = float(max(retrieval_scores) - retrieval_scores[K - 1])
        probs = _softmax(list(retrieval_scores[:K]), tau=float(getattr(s, "judge_entropy_tau", 0.15)))
        entropy = _entropy(probs)

    # Dense vs Lexical Jaccard on top-50
    jaccard = 0.0
    if dense_ids and lex_ids:
        a = set(list(dense_ids)[:50])
        b = set(list(lex_ids)[:50])
        inter = len(a & b)
        union = len(a | b) or 1
        jaccard = float(inter) / float(union)

    # Budget/category/dup ratios on top-k
    tol = float(getattr(s, "judge_tolerance_budget", 0.05))
    mx = getattr(getattr(facets, "budget_usd", None), "max", None)
    include_cats = [str(x).lower() for x in (getattr(facets, "categories_include", None) or [])]
    def _cat_ok(it: Dict[str, Any]) -> bool:
        if not include_cats:
            return True
        cats = it.get("categories")
        if isinstance(cats, list):
            cats_str = " ".join(str(x).lower() for x in cats)
        elif hasattr(cats, "tolist"):
            try:
                arr = cats.tolist()
                cats_str = " ".join(str(x).lower() for x in (arr if isinstance(arr, list) else [arr]))
            except Exception:
                cats_str = str(cats).lower()
        else:
            cats_str = str(cats).lower()
        return any(c in cats_str for c in include_cats)

    head = fused_candidates[:k]
    budget_ok = 0
    cat_ok = 0
    stems: dict[str, int] = {}
    asins: dict[str, int] = {}
    for it in head:
        price = it.get("price")
        if isinstance(mx, (int, float)) and isinstance(price, (int, float)) and price <= mx * (1.0 + tol):
            budget_ok += 1
        if _cat_ok(it):
            cat_ok += 1
        asin = str(it.get("parent_asin") or "")
        if asin:
            asins[asin] = asins.get(asin, 0) + 1
        stem = it.get("title_stem") or _norm_title_stem(it.get("title"))
        if stem:
            stems[stem] = stems.get(stem, 0) + 1
    budget_ok_ratio = (budget_ok / float(k)) if k > 0 else 0.0
    category_ok_ratio = (cat_ok / float(k)) if k > 0 else 0.0
    dup_count = 0
    for cnt in asins.values():
        if cnt > 1:
            dup_count += cnt - 1
    for cnt in stems.values():
        if cnt > 1:
            dup_count += cnt - 1
    dup_ratio = (dup_count / float(k)) if k > 0 else 0.0

    return {
        "specificity": float(specificity),
        "margin": float(margin),
        "entropy": float(entropy),
        "dense_lex_jaccard@50": float(jaccard),
        "budget_ok@20": float(budget_ok_ratio),
        "category_ok@20": float(category_ok_ratio),
        "dup@20": float(dup_ratio),
    }


def decide_mode(signals: Dict[str, float], system: Dict[str, Any]) -> Dict[str, Any]:
    s = get_settings()
    spec = signals.get("specificity", 0.0)
    margin = signals.get("margin", 0.0)
    entropy = signals.get("entropy", 0.0)
    jacc = signals.get("dense_lex_jaccard@50", 0.0)
    bud = signals.get("budget_ok@20", 0.0)
    cat = signals.get("category_ok@20", 0.0)
    dup = signals.get("dup@20", 0.0)

    if system.get("circuit_open") or system.get("cost_guardrail_hit"):
        return {"mode": "skip", "top_m": 0, "reason": "circuit_or_cost"}

    # SKIP
    if (
        spec >= 3
        and margin >= float(getattr(s, "judge_margin_skip_min", 0.05))
        and entropy <= float(getattr(s, "judge_entropy_skip_max", 1.2))
        and jacc >= float(getattr(s, "judge_jaccard_skip_min", 0.6))
        and bud >= float(getattr(s, "judge_budget_ok_skip_min", 0.8))
        and cat >= float(getattr(s, "judge_category_ok_skip_min", 0.7))
        and dup <= float(getattr(s, "judge_dup_skip_max", 0.25))
    ):
        return {"mode": "skip", "top_m": 0, "reason": "high_conf_retrieval"}

    # CHEAP borderline
    cheap = False
    if spec in {1.0, 2.0}:
        cheap = True
    if (1.2 < entropy <= 1.6) or (0.02 <= margin < 0.05):
        cheap = True
    if 0.35 < jacc < 0.6:
        cheap = True
    if (0.6 <= bud < 0.8) or (0.25 < dup <= 0.4):
        cheap = True
    if cheap:
        return {"mode": "cheap", "top_m": int(getattr(s, "judge_cheap_top_m", 12)), "reason": "borderline"}

    # FULL otherwise
    return {"mode": "full", "top_m": int(getattr(s, "judge_full_top_m", 24)), "reason": "low_conf_or_violations"}


def post_judge_quality_check(facets: Any, judged_items: List[Dict[str, Any]]) -> bool:
    s = get_settings()
    tol = float(getattr(s, "judge_tolerance_budget", 0.05))
    mx = getattr(getattr(facets, "budget_usd", None), "max", None)
    k = min(20, len(judged_items))
    head = judged_items[:k]
    budget_ok = 0
    stems: dict[str, int] = {}
    asins: dict[str, int] = {}
    for it in head:
        price = it.get("price")
        if isinstance(mx, (int, float)) and isinstance(price, (int, float)) and price <= mx * (1.0 + tol):
            budget_ok += 1
        asin = str(it.get("parent_asin") or "")
        if asin:
            asins[asin] = asins.get(asin, 0) + 1
        stem = it.get("title_stem") or _norm_title_stem(it.get("title"))
        if stem:
            stems[stem] = stems.get(stem, 0) + 1
    budget_ratio = (budget_ok / float(k)) if k > 0 else 0.0
    dup_count = 0
    for cnt in asins.values():
        if cnt > 1:
            dup_count += cnt - 1
    for cnt in stems.values():
        if cnt > 1:
            dup_count += cnt - 1
    dup_ratio = (dup_count / float(k)) if k > 0 else 0.0
    if budget_ratio < float(getattr(s, "judge_budget_ok_skip_min", 0.8)):
        return True
    if dup_ratio > float(getattr(s, "judge_dup_skip_max", 0.25)):
        return True
    return False


def projected_cost_ok(
    model: str,
    top_m: int,
    avg_tokens_per_item: int,
    per_req_cap: float,
    day_cap: float,
) -> bool:
    # Estimate prompt+completion roughly from items; completion small/ignored in cheap mode
    predicted_tokens = int(top_m) * int(max(1, avg_tokens_per_item))
    # Get unit cost from telemetry estimator
    est_cost_usd = telemetry_counters.estimate_model_cost(model, predicted_tokens, 0)
    if est_cost_usd > float(per_req_cap):
        return False
    snap = telemetry_counters.snapshot()
    spent_today = 0.0
    for m, row in (snap.get("models", {}) or {}).items():
        spent_today += float(row.get("cost_usd_est", 0.0) or 0.0)
    if (spent_today + est_cost_usd) > float(day_cap):
        return False
    return True


def fingerprint(model_id: str, rubric_version: Any, normalized_query: str, fused_topM_ids: Sequence[str], index_signature: str | None = None) -> str:
    sig = index_signature or "nosig"
    key = f"{model_id}|{str(rubric_version)}|{normalized_query}|{sig}|{','.join(map(str, fused_topM_ids))}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"sha1:{h}"


