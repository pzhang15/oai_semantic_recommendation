from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.core.embedding import embed_query
from src.core.faiss_store import FaissStore
from src.core.config import get_settings
from src.core.fusion import rrf_fuse


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
    # Load a richer set of columns if available to power detailed item views/tests
    cols = [
        "id",
        "title",
        "brand",
        "price",
        "image_url",
        "product_url",
        # extended metadata (if present in parquet)
        "main_category",
        "average_rating",
        "rating_number",
        "features",
        "description",
        "description_list",
        "images",
        "videos",
        "store",
        "categories",
        "details",
        "parent_asin",
        "bought_together",
    ]
    # Some parquet writers may omit a subset; select only existing columns
    all_cols = pd.read_parquet(parquet_path, columns=None).columns.tolist()
    use_cols = [c for c in cols if c in all_cols]
    table = pd.read_parquet(parquet_path, columns=use_cols)
    return table


def _compose_query_from_facets(query_text: str, facets: Any | None) -> str:
    txt = (query_text or "").strip()
    if not facets:
        return txt
    parts: list[str] = []
    for arr in [getattr(facets, "colors", None) or [], getattr(facets, "materials", None) or [], getattr(facets, "categories_include", None) or []]:
        for t in arr:
            s = str(t).strip().lower()
            if s and s not in parts:
                parts.append(s)
    if parts:
        extra = " ".join(parts[:30])
        txt = f"{txt} {extra}" if txt else extra
    return txt


def retrieve(text: str, k: int | None = None, facets: Any | None = None) -> Dict[str, Any]:
    if _store is None:
        # Lazy-load store for script usage (when FastAPI startup hasn't run)
        settings = get_settings()
        store = FaissStore()
        store.load(index_path=settings.index_path, ids_path=settings.ids_path, stats_path=settings.stats_path)
        set_store(store)
    settings = get_settings()
    top_k = int(k or settings.top_k_default)

    t0 = time.perf_counter()
    q = embed_query(text)
    t1 = time.perf_counter()
    dense_k = max(top_k, settings.hybrid_dense_k if settings.hybrid_enabled else top_k)
    faiss_span = 0.0

    def _faiss_search() -> tuple[List[int], List[float], float]:
        s0 = time.perf_counter()
        ii, ss = _store.search(q, dense_k)
        s1 = time.perf_counter()
        return ii, ss, (s1 - s0)

    # Load ids list once; map to top_ids after search completes
    ids = _load_ids(settings.ids_path)

    # Optional lexical + fusion
    hybrid_trace: Dict[str, Any] = {
        "enabled": False,
    }
    fused_ids: List[str] = []
    lex_hits: List[Dict[str, Any]] = []
    if settings.hybrid_enabled:
        try:
            from src.core.lexical import search_lexical
            lex_k = max(top_k, settings.hybrid_lex_k)
            qtext = _compose_query_from_facets(text, facets)
            # Run FAISS and lexical concurrently
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_dense = ex.submit(_faiss_search)
                fut_lex = ex.submit(search_lexical, qtext, lex_k, facets, True)
                (idxs, scores, faiss_span) = fut_dense.result()
                lex_res = fut_lex.result()
            if isinstance(lex_res, tuple):
                lex_hits, lex_times = lex_res
            else:
                lex_hits, lex_times = lex_res, {}
            t3 = time.perf_counter()
            lex_ids = [h["id"] for h in lex_hits]
            t4a = time.perf_counter()
            # recompute top_ids based on possibly new idxs
            top_ids = [ids[i] for i in idxs if i >= 0 and i < len(ids)]
            fused_ids = rrf_fuse({"dense": top_ids, "lex": lex_ids}, k0=settings.hybrid_rrf_k, topn=max(top_k, len(top_ids)))
            t4b = time.perf_counter()
            hybrid_trace = {
                "enabled": True,
                "rrf_k": settings.hybrid_rrf_k,
                "dense_k": dense_k,
                "lex_k": lex_k,
                "dense_hits": len(top_ids),
                "lex_hits": len(lex_ids),
                "fused_unique": len(set(fused_ids)),
                "dense_ids": top_ids[:max(top_k, 200)],
                "lex_ids": lex_ids[:max(top_k, 200)],
                "timings": {
                    "embed_ms": round((t1 - t0) * 1000.0, 2),
                    "faiss_ms": round(faiss_span * 1000.0, 2),
                    **({k: float(v) for k, v in lex_times.items()} if lex_times else {}),
                    "fuse_ms": round((t4b - t4a) * 1000.0, 2),
                }
            }
        except Exception:
            hybrid_trace = {"enabled": False}
            # dense-only fallback
            idxs, scores, faiss_span = _faiss_search()
            top_ids = [ids[i] for i in idxs if i >= 0 and i < len(ids)]
    else:
        # dense-only path
        idxs, scores, faiss_span = _faiss_search()
        top_ids = [ids[i] for i in idxs if i >= 0 and i < len(ids)]
    # Load product table (cached) and filter to found ids
    t5 = time.perf_counter()
    df = load_products_table(settings.parquet_path)
    # Choose ordering source: fused if hybrid, else dense
    order_ids = fused_ids if hybrid_trace.get("enabled") else top_ids
    sub = df[df["id"].isin(order_ids)].copy()

    # Preserve FAISS order (stable)
    order = {pid: j for j, pid in enumerate(order_ids)}
    sub["__order"] = sub["id"].map(order)
    sub = sub.sort_values("__order").drop(columns="__order")

    # Build items list with scores
    score_map = {pid: float(sc) for pid, sc in zip(top_ids, scores)}
    items: List[Dict[str, Any]] = []

    def infer_brand_from_title(title: str | None) -> str | None:
        if not title:
            return None
        t = str(title)
        for marker in [" Men's", " Men's ", " men'", " Women's", " women's", " - ", " – ", ": "]:
            idx = t.find(marker)
            if idx > 0:
                cand = t[:idx].strip()
                if 2 <= len(cand) <= 80:
                    return cand
        # fallback: first 2 words if capitalized
        parts = t.split()
        if parts and parts[0][0:1].isupper():
            head = parts[0]
            if len(parts) > 1 and parts[1][0:1].isupper():
                head += " " + parts[1]
            return head
        return None
    for j, row in enumerate(sub.itertuples(index=False)):
        pid = getattr(row, "id")
        pid_str = str(pid)
        title = getattr(row, "title", None)
        brand = getattr(row, "brand", None) or infer_brand_from_title(title)
        price = getattr(row, "price", None)
        image_url = getattr(row, "image_url", None)
        product_url = getattr(row, "product_url", None)
        sc = score_map.get(pid_str, None)
        # Ensure numeric price or None
        if isinstance(price, float) and (not np.isfinite(price)):
            price_val: float | None = None
        else:
            price_val = float(price) if isinstance(price, (int, float)) else None
        # Extended metadata (best-effort if present)
        main_category = getattr(row, "main_category", None)
        average_rating = getattr(row, "average_rating", None)
        rating_number = getattr(row, "rating_number", None)
        features = getattr(row, "features", None)
        description = getattr(row, "description", None)
        description_list = getattr(row, "description_list", None)
        images = getattr(row, "images", None)
        videos = getattr(row, "videos", None)
        store = getattr(row, "store", None)
        categories = getattr(row, "categories", None)
        details = getattr(row, "details", None)
        parent_asin = getattr(row, "parent_asin", None)
        bought_together = getattr(row, "bought_together", None)

        # Prefer list description if string description is missing; normalize description to list if possible
        desc_value = description_list if description_list is not None else description

        item: Dict[str, Any] = {
            "id": pid_str,
            "brand": brand,
            "price": price_val,
            "image_url": image_url,
            "product_url": product_url,
            "score": float(sc) if sc is not None else 0.0,
            "raw_score": float(sc) if sc is not None else 0.0,
            "vector_index": idxs[j] if j < len(idxs) else None,
        }
        # Only include title if present; frontend schema expects string when provided (not null)
        if title is not None:
            item["title"] = title
        # Attach extended keys if present in dataframe
        if "main_category" in sub.columns:
            item["main_category"] = main_category
        if "average_rating" in sub.columns:
            item["average_rating"] = average_rating
        if "rating_number" in sub.columns:
            item["rating_number"] = rating_number
        if "features" in sub.columns:
            item["features"] = features
        if ("description" in sub.columns) or ("description_list" in sub.columns):
            item["description"] = desc_value
        if "images" in sub.columns:
            item["images"] = images
        if "videos" in sub.columns:
            item["videos"] = videos
        if "store" in sub.columns:
            item["store"] = store
        if "categories" in sub.columns:
            item["categories"] = categories
        if "details" in sub.columns:
            item["details"] = details
        if "parent_asin" in sub.columns:
            item["parent_asin"] = parent_asin
        if "bought_together" in sub.columns:
            item["bought_together"] = bought_together

        items.append(item)

    # Prefer items that have richer metadata (price/image/brand) while preserving relative order otherwise
    def quality_key(it: Dict[str, Any]) -> tuple[int, int]:
        q = int(it.get("price") is not None) + int(it.get("image_url") is not None) + int(it.get("brand") not in (None, ""))
        return (q, 0)
    items = sorted(items, key=quality_key, reverse=True)

    # Optionally drop items missing both price and image to improve UX
    if settings.search_require_price_or_image:
        items = [it for it in items if (it.get("price") is not None) or (it.get("image_url") not in (None, ""))]

    t6 = time.perf_counter()
    return {
        "items": items[:top_k],
        "trace": {
            "k": top_k,
            "dim": _store.dimension,
            "ntotal": _store.size,
            "hybrid": hybrid_trace,
            "timings": {
                "embed_ms": round((t1 - t0) * 1000.0, 2),
                "faiss_ms": round(faiss_span * 1000.0, 2),
                "join_ms": round((t6 - t5) * 1000.0, 2),
            },
        },
    }


