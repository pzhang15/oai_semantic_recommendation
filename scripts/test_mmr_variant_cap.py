import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.mmr import mmr_select  # type: ignore


def main() -> None:
    items = []
    for i in range(8):
        items.append({
            "id": f"dom{i}",
            "title": f"BrandX Essential Tee Black {i}",
            "brand": "BrandX",
            "score": 0.9 - 0.01*i,
        })
    for i in range(8):
        items.append({
            "id": f"uni{i}",
            "title": f"BrandY Alt Shirt {i}",
            "brand": "BrandY",
            "score": 0.7 - 0.01*i,
        })
    sel, trace = mmr_select(items, lambda_mult=0.7, final_k=12, mode="title_only")
    stems = {}
    for it in sel:
        stem = it.get("title_stem")
        stems[stem] = stems.get(stem, 0) + 1
    if any(c > 2 for c in stems.values()):
        print("[FAIL] more than 2 from same variant group")
        raise SystemExit(1)
    print("[OK] mmr_variant_cap")


if __name__ == "__main__":
    main()


