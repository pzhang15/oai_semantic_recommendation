import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.parser import parse_query  # type: ignore
from src.core.retriever import retrieve  # type: ignore
from src.core.rerank import judge_rerank  # type: ignore


def main() -> None:
    facets = parse_query("linen shirt under $60")
    cands = retrieve("linen shirt under $60", k=20).get("items", [])[:8]
    out4, _ = judge_rerank(facets, cands, top_m=8, batch_size=4)
    out8, _ = judge_rerank(facets, cands, top_m=8, batch_size=8)
    if not out4 or not out8:
        print("[FAIL] empty outputs")
        raise SystemExit(1)
    if out4[0].get("id") != out8[0].get("id"):
        print("[WARN] top-1 differs (tolerated)")
    top5_a = {it.get("id") for it in out4[:5]}
    top5_b = {it.get("id") for it in out8[:5]}
    overlap = len(top5_a & top5_b)
    if overlap < 3:  # >= 60% overlap
        print("[FAIL] batch consistency too low")
        raise SystemExit(1)
    print("[OK] judge_batch_consistency")


if __name__ == "__main__":
    main()


