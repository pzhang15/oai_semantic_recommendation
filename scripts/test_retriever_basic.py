import json
import os
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'src' is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.embedding import embed_query
from src.core.faiss_store import FaissStore


def main() -> None:
    settings = get_settings()

    store = FaissStore()
    store.load(
        index_path=settings.index_path,
        ids_path=settings.ids_path,
        stats_path=settings.stats_path,
    )

    q = embed_query("linen shirt for summer")
    idxs, scores = store.search(q, 5)
    if len(idxs) != 5 or len(scores) != 5:
        print("[ERROR] Expected 5 results")
        raise SystemExit(1)
    if not np.all(np.isfinite(scores)):
        print("[ERROR] Non-finite scores returned")
        raise SystemExit(1)

    with Path(settings.ids_path).open("r", encoding="utf-8") as f:
        ids = json.load(f)
    top_ids = [ids[i] for i in idxs if i >= 0 and i < len(ids)]

    df = pd.read_parquet(settings.parquet_path, columns=["id", "title"])
    sub = df[df["id"].isin(top_ids)]
    if len(sub) < 5:
        print("[ERROR] Some product IDs not found in parquet")
        raise SystemExit(1)
    if sub["id"].duplicated().any():
        print("[ERROR] Duplicate IDs found in results")
        raise SystemExit(1)
    if (sub["title"].astype(str).str.strip() == "").any():
        print("[ERROR] Empty titles detected")
        raise SystemExit(1)

    print("[OK] Retriever basic: 5 results, scores finite, IDs present, titles non-empty")


if __name__ == "__main__":
    main()


