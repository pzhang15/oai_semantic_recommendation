import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.parser import parse_query  # type: ignore
from src.core.retriever import retrieve  # type: ignore
from src.core.rerank import judge_rerank, set_test_mode  # type: ignore


def main() -> None:
    facets = parse_query("linen shirt under $60")
    cands = retrieve("linen shirt under $60", k=12).get("items", [])[:6]
    try:
        set_test_mode(temp_reject=True)
        out, trace = judge_rerank(facets, cands, top_m=6, batch_size=3)
    finally:
        set_test_mode(temp_reject=False)
    if not out or len(out) != len(cands):
        print("[FAIL] fallback produced empty output")
        raise SystemExit(1)
    print("[OK] judge_fallback")


if __name__ == "__main__":
    main()


