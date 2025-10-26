import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.mmr import mmr_select  # type: ignore


def main() -> None:
    items = []
    for i in range(6):
        items.append({"id": f"a{i}", "title": f"black cotton t-shirt colorway {i}", "score": 0.9 - 0.01*i})
    for i in range(6):
        items.append({"id": f"b{i}", "title": f"navy linen shirt variant {i}", "score": 0.85 - 0.01*i})
    sel, trace = mmr_select(items, lambda_mult=0.7, final_k=8, mode="title_only", title_threshold=0.85)
    stems = {}
    for it in sel:
        stem = it.get("title_stem") or ""
        stems[stem] = stems.get(stem, 0) + 1
    if any(c > 2 for c in stems.values()):
        print("[FAIL] too many from same title stem")
        raise SystemExit(1)
    print("[OK] mmr_title_fallback")


if __name__ == "__main__":
    main()


