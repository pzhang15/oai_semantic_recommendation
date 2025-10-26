import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.parser import parse_query  # type: ignore
from src.core.retriever import retrieve  # type: ignore
from src.core.rerank import judge_rerank, get_judge_cache_stats  # type: ignore


def main() -> None:
    facets = parse_query("linen shirt under $60")
    cands = retrieve("linen shirt under $60", k=16).get("items", [])[:8]
    judge_rerank(facets, cands, top_m=8, batch_size=4)
    before = get_judge_cache_stats()["hits"]
    judge_rerank(facets, cands, top_m=8, batch_size=4)
    after = get_judge_cache_stats()["hits"]
    if after <= before:
        print("[FAIL] cache hits did not increase")
        raise SystemExit(1)
    print("[OK] judge_cache")


if __name__ == "__main__":
    main()


