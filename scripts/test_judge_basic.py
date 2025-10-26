import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings  # type: ignore
from src.core.retriever import retrieve  # type: ignore
from src.core.parser import parse_query  # type: ignore
from src.core.rerank import judge_rerank  # type: ignore


def main() -> None:
    facets = parse_query("linen shirt under $60")
    ret = retrieve("linen shirt under $60", k=20)
    cands = ret.get("items", [])[:8]
    out, trace = judge_rerank(facets, cands, top_m=8, batch_size=4)
    if len(out) != len(cands):
        print("[FAIL] length mismatch")
        raise SystemExit(1)
    if any((it.get("judge_score") is None or not (0.0 <= float(it.get("judge_score")) <= 1.0)) for it in out):
        print("[FAIL] judge_score out of range")
        raise SystemExit(1)
    if any((float(it.get("score") or 0.0) < 0) for it in out):
        print("[FAIL] negative combined score")
        raise SystemExit(1)
    def has_reason(it: dict) -> bool:
        w = (it.get("why") or "").lower()
        return ("linen" in w) or ("light" in w) or ("budget" in w)
    if not any(has_reason(it) for it in out):
        print("[FAIL] no rationale mentions linen/light/budget")
        raise SystemExit(1)
    print("[OK] judge_basic")


if __name__ == "__main__":
    main()


