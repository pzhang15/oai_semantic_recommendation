from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core.embedding import embed_query
from src.core.faiss_store import FaissStore
from src.core.config import get_settings


_store: FaissStore | None = None


def set_store(store: FaissStore) -> None:
    global _store
    _store = store


def get_store() -> FaissStore | None:
    return _store


@lru_cache
def _load_ids(ids_path: str) -> List[str]:
    with Path(ids_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ids.json is not a list")
    return [str(x) for x in data]


@lru_cache
def load_products_table(parquet_path: str) -> pd.DataFrame:
    cols = ["id", "title", "brand", "price", "image_url", "product_url"]
    table = pd.read_parquet(parquet_path, columns=cols)
    return table


def retrieve(text: str, k: int | None = None) -> Dict[str, Any]:
    if _store is None:
        # Lazy-load store for script usage (when FastAPI startup hasn't run)
        settings = get_settings()
        store = FaissStore()
        store.load(index_path=settings.index_path, ids_path=settings.ids_path, stats_path=settings.stats_path)
        set_store(store)
    settings = get_settings()
    top_k = int(k or settings.top_k_default)

    q = embed_query(text)
    idxs, scores = _store.search(q, top_k)

    # Map row indices -> ids via ids.json
    ids = _load_ids(settings.ids_path)
    top_ids = [ids[i] for i in idxs if i >= 0 and i < len(ids)]

    # Load product table (cached) and filter to found ids
    df = load_products_table(settings.parquet_path)
    sub = df[df["id"].isin(top_ids)].copy()

    # Preserve FAISS order (stable)
    order = {pid: j for j, pid in enumerate(top_ids)}
    sub["__order"] = sub["id"].map(order)
    sub = sub.sort_values("__order").drop(columns="__order")

    # Build items list with scores
    score_map = {pid: float(sc) for pid, sc in zip(top_ids, scores)}
    items: List[Dict[str, Any]] = []
    for j, row in enumerate(sub.itertuples(index=False)):
        pid = getattr(row, "id")
        title = getattr(row, "title", None)
        brand = getattr(row, "brand", None)
        price = getattr(row, "price", None)
        image_url = getattr(row, "image_url", None)
        product_url = getattr(row, "product_url", None)
        sc = score_map.get(pid, None)
        # Ensure numeric price or None
        if isinstance(price, float) and (not np.isfinite(price)):
            price_val: float | None = None
        else:
            price_val = float(price) if isinstance(price, (int, float)) else None
        items.append({
            "id": pid,
            "title": title,
            "brand": brand,
            "price": price_val,
            "image_url": image_url,
            "product_url": product_url,
            "score": float(sc) if sc is not None else None,
            "raw_score": float(sc) if sc is not None else None,
            "vector_index": idxs[j] if j < len(idxs) else None,
        })

    return {
        "items": items[:top_k],
        "trace": {
            "k": top_k,
            "dim": _store.dimension,
            "ntotal": _store.size,
        },
    }


