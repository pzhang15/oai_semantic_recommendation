# scripts/smoke_test.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

# FAISS + OpenAI
import faiss  # conda-forge build on Windows recommended
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype("float32", copy=False)
    n = np.linalg.norm(vec, ord=2)
    if not np.isfinite(n) or n == 0.0:
        return vec
    return vec / (n + 1e-12)


def load_ids(ids_path: Path) -> List[str]:
    try:
        with ids_path.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        if not isinstance(ids, list):
            die(f"ids.json is not a list: {ids_path}")
        return [str(x) for x in ids]
    except FileNotFoundError:
        die(f"ids.json not found at {ids_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test: embed query → FAISS search → show top results")
    ap.add_argument("--query", required=True, help="Natural-language query to test")
    ap.add_argument("--k", type=int, default=10, help="Top-K results to return")
    ap.add_argument("--index-path", default="data/index.faiss", help="Path to FAISS index")
    ap.add_argument("--ids-path", default="data/ids.json", help="Path to row→id mapping (list)")
    ap.add_argument("--stats-path", default="data/stats.json", help="Path to stats.json (for dim/model)")
    ap.add_argument("--parquet-path", default="data/products.parquet", help="Path to normalized product parquet")
    ap.add_argument("--model-embed", default=os.getenv("MODEL_EMBED", "text-embedding-3-small"),
                    help="Embedding model to use")
    args = ap.parse_args()

    idx_path = Path(args.index_path)
    ids_path = Path(args.ids_path)
    stats_path = Path(args.stats_path)
    parquet_path = Path(args.parquet_path)

    # ---------- Load artifacts ----------
    if not idx_path.exists():
        die(f"FAISS index not found: {idx_path}")
    try:
        index = faiss.read_index(str(idx_path))
    except Exception as e:
        die(f"Failed to read FAISS index: {e}")

    ids = load_ids(ids_path)
    ntotal = index.ntotal
    if ntotal != len(ids):
        die(f"Index size ({ntotal}) != ids.json length ({len(ids)}). Make sure they were built together.")

    # stats (optional but nice)
    dim_expected = None
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        dim_expected = int(stats.get("dim", 0)) or None
        print(f"[INFO] stats.json: model={stats.get('model')} dim={stats.get('dim')} rows={stats.get('rows')}")

    print(f"[INFO] faiss index: type=flatip? dim={index.d} ntotal={ntotal}")
    if dim_expected and dim_expected != index.d:
        die(f"Dimension mismatch: stats.json dim={dim_expected} vs index.d={index.d}")

    # ---------- Embed the query ----------
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    if not api_key:
        die("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        emb = client.embeddings.create(model=args.model_embed, input=args.query).data[0].embedding
    except Exception as e:
        die(f"OpenAI embeddings call failed: {e}")

    q = np.asarray(emb, dtype="float32")
    q = l2_normalize(q)[None, :]  # shape (1, d)

    if q.shape[1] != index.d:
        die(f"Query dim {q.shape[1]} != index dim {index.d}. Wrong embedding model?")

    # ---------- Search ----------
    topk = max(1, min(int(args.k), ntotal))
    D, I = index.search(q, topk)  # I: row indices; D: similarities
    idxs = I[0].tolist()
    scores = D[0].tolist()

    # ---------- Load product table & map results ----------
    if not parquet_path.exists():
        die(f"Parquet not found: {parquet_path}")

    # Load only the columns we need
    table = pq.read_table(parquet_path, columns=["id", "title", "brand", "price", "image_url", "product_url"])
    df = table.to_pandas()

    # Map row indices → ids.json → product rows; keep order
    top_ids = [ids[i] for i in idxs]
    sub = df[df["id"].isin(top_ids)].copy()

    # Reorder to match search order
    order = {pid: j for j, pid in enumerate(top_ids)}
    sub["__order"] = sub["id"].map(order)
    sub = sub.sort_values("__order").drop(columns="__order")

    # ---------- Print summary ----------
    print("\n=== Smoke Test Summary ===")
    print(f"Query: {args.query}")
    print(f"Top-K: {topk}")
    print(f"Index: {idx_path} (dim={index.d}, ntotal={ntotal})")
    print()

    # Pretty print results
    if sub.empty:
        print("[WARN] No rows matched by id in products.parquet (check that ids.json and parquet align).")
    else:
        for j, row in enumerate(sub.itertuples(index=False), start=1):
            pid = getattr(row, "id")
            title = getattr(row, "title")
            brand = getattr(row, "brand")
            price = getattr(row, "price")
            url = getattr(row, "product_url")
            sc = scores[j - 1] if j - 1 < len(scores) else None
            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else (price or "")
            print(f"{j:2d}. id={pid}  score={sc:.4f}  brand={brand or ''}  price={price_str}")
            print(f"    {title}")
            if url:
                print(f"    url: {url}")
        print()

    print("[OK] Smoke test completed.")


if __name__ == "__main__":
    main()
