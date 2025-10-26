import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.mmr import mmr_select  # type: ignore


def main() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=1536).astype(np.float32)
    base /= np.linalg.norm(base) + 1e-12
    close = [base + 0.001 * rng.normal(size=1536).astype(np.float32) for _ in range(3)]
    close = [v / (np.linalg.norm(v) + 1e-12) for v in close]
    far = [rng.normal(size=1536).astype(np.float32) for _ in range(3)]
    far = [v / (np.linalg.norm(v) + 1e-12) for v in far]

    items = []
    for i, v in enumerate(close):
        items.append({"id": f"c{i}", "title": f"black tee {i}", "score": 0.9, "vector": v, "vector_index": i})
    for i, v in enumerate(far):
        items.append({"id": f"d{i}", "title": f"diverse item {i}", "score": 0.8, "vector": v, "vector_index": 100+i})

    sel, trace = mmr_select(items, lambda_mult=0.7, final_k=4, mode="vector_only")
    ids = [it["id"] for it in sel]
    if all(x.startswith("c") for x in ids[:3]):
        print("[FAIL] near-duplicates dominated selection")
        raise SystemExit(1)
    print("[OK] mmr_vector_cosine")


if __name__ == "__main__":
    main()


