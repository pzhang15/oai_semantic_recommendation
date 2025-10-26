import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.schemas import QueryFacets, Budget  # type: ignore
from src.core.rerank import judge_rerank  # type: ignore


def main() -> None:
    facets = QueryFacets(
        intent="search",
        budget_usd=Budget(min=None, max=60, currency="USD"),
        num_items=12,
    )
    # Synthetic candidates with equal retrieval
    cands = [
        {"id": "a", "title": "basic tee", "brand": "x", "price": 40.0, "raw_score": 0.5},
        {"id": "b", "title": "basic tee", "brand": "y", "price": 55.0, "raw_score": 0.5},
        {"id": "c", "title": "basic tee", "brand": "z", "price": 120.0, "raw_score": 0.5},
    ]
    out, trace = judge_rerank(facets, cands, top_m=3, batch_size=3)
    # Expect the $120 item to rank last and have low judge_score
    ids_order = [it.get("id") for it in out]
    if ids_order[-1] != "c":
        print("[FAIL] over-budget item not last")
        raise SystemExit(1)
    over = next((it for it in out if it.get("id") == "c"), None)
    if over is None or float(over.get("judge_score") or 0.0) >= 0.5:
        print("[FAIL] over-budget item not penalized")
        raise SystemExit(1)
    print("[OK] judge_budget_penalty")


if __name__ == "__main__":
    main()


