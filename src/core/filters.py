from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.models.schemas import QueryFacets


def apply_filters(
    candidates: List[Dict[str, Any]],
    facets: QueryFacets,
    *,
    price_tolerance: float = 0.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    reasons: Dict[str, int] = {}

    def drop(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    kept: List[Dict[str, Any]] = []
    max_price = facets.budget_usd.max if facets and facets.budget_usd else None
    max_price_tol = (max_price * (1.0 + price_tolerance)) if isinstance(max_price, (int, float)) else None
    cats_excl = set((facets.categories_exclude or []))
    gender = (facets.gender_or_fit or "").lower() if facets else ""
    hard = [h.lower() for h in (facets.hard_constraints or [])]

    for it in candidates:
        title = (it.get("title") or "").lower()
        price = it.get("price")
        cat_hit_excl = False
        for c in cats_excl:
            if c and c in title:
                cat_hit_excl = True
                break
        if cat_hit_excl:
            drop("category_excluded")
            continue

        if isinstance(max_price_tol, (int, float)) and isinstance(price, (int, float)):
            if np.isfinite(price) and price > max_price_tol:
                drop("over_budget")
                continue

        if gender in {"men", "women"}:
            if gender == "men" and ("women" in title or "women's" in title or "womens" in title or "ladies" in title):
                drop("gender_mismatch")
                continue
            if gender == "women" and ("men" in title or "men's" in title or "mens" in title):
                drop("gender_mismatch")
                continue

        if hard:
            violated = False
            for h in hard:
                # Only simple negations like 'no logos' or 'avoid polyester'
                key = h.replace("no ", "").replace("avoid ", "").replace("without ", "").replace("not ", "").strip()
                if key and key in title:
                    violated = True
                    break
            if violated:
                drop("hard_constraint")
                continue

        kept.append(it)

    return kept, reasons


