import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import time
from typing import Tuple
from openai import OpenAI

# Allow running this script directly (so 'src' is importable)
try:
    from src.core.config import get_settings
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from src.core.config import get_settings


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _flatten_list(values: Any) -> List[str]:
    result: List[str] = []
    if values is None:
        return result
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, list):
        for v in values:
            result.extend(_flatten_list(v))
    return [v for v in (x.strip() for x in result) if v]


def _parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace("$", "").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _pick_first(*keys: str, from_obj: Dict[str, Any]) -> Any:
    for k in keys:
        if k in from_obj and from_obj[k] not in (None, ""):
            return from_obj[k]
    return None


def _build_text_for_embedding(title: str, brand: str, categories: List[str], features: List[str], description: str, max_len: int = 1500) -> str:
    segments: List[str] = []
    if title:
        segments.append(title)
    if brand:
        segments.append(brand)
    if categories:
        segments.append(" | ".join(dict.fromkeys(categories)))
    if features:
        segments.append(" | ".join(dict.fromkeys(features)))
    if description:
        segments.append(description)
    text = ". ".join(s for s in segments if s)
    return text[:max_len]


def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"id": None}

    asin = _to_string(
        _pick_first(
            "id",
            "asin",
            "ASIN",
            "parent_asin",
            "item_id",
            "product_id",
            "sku",
            from_obj=raw,
        )
    )
    title = _to_string(_pick_first("title", from_obj=raw))
    brand = _to_string(_pick_first("brand", from_obj=raw))

    categories = _flatten_list(_pick_first("categories", from_obj=raw))
    features = _flatten_list(_pick_first("feature", "features", from_obj=raw))

    desc_val = _pick_first("description", from_obj=raw)
    if isinstance(desc_val, list):
        description = " ".join(_flatten_list(desc_val))
    else:
        description = _to_string(desc_val)

    price = _parse_price(_pick_first("price", from_obj=raw))

    rating = None
    rating_src = _pick_first("rating", "average_rating", "overall", from_obj=raw)
    if isinstance(rating_src, (int, float)):
        rating = float(rating_src)
    elif isinstance(rating_src, str):
        try:
            rating = float(rating_src)
        except ValueError:
            rating = None

    rating_count = None
    rc_src = _pick_first("rating_count", "review_count", "vote", "ratings_count", from_obj=raw)
    if isinstance(rc_src, (int, float)):
        rating_count = int(rc_src)
    elif isinstance(rc_src, str):
        try:
            rating_count = int(float(rc_src))
        except ValueError:
            rating_count = None

    image_url = None
    img_src = _pick_first(
        "imUrl",
        "imageURLHighRes",
        "imageURL",
        "image",
        "image_url",
        from_obj=raw,
    )
    if isinstance(img_src, list):
        image_url = _to_string(img_src[0] if img_src else None)
    else:
        image_url = _to_string(img_src)

    product_url = _to_string(_pick_first("product_url", "url", from_obj=raw))

    text_for_embedding = _build_text_for_embedding(
        title=title,
        brand=brand,
        categories=categories,
        features=features,
        description=description,
    )

    return {
        "id": asin or None,
        "asin": asin or None,
        "title": title or None,
        "brand": brand or None,
        "categories": categories,
        "features": features,
        "description": description or None,
        "price": price,
        "rating": rating,
        "rating_count": rating_count,
        "image_url": image_url or None,
        "product_url": product_url or None,
        "text_for_embedding": text_for_embedding or None,
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream records from a JSONL file or a JSON array-of-objects file.

    - Skips empty lines and bracket/comma separators
    - Trims trailing commas on single-line objects within a JSON array
    """
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # Skip array delimiters and separators
            if line in ("[", "]", "],", ","):
                continue
            if line.endswith(","):
                line = line[:-1].rstrip()
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # If objects span multiple lines, they won't parse here; skip silently
                # Dataset like Amazon meta usually has one object per line
                continue


def _to_table(records: List[Dict[str, Any]]):
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("asin", pa.string()),
            pa.field("title", pa.string()),
            pa.field("brand", pa.string()),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("features", pa.list_(pa.string())),
            pa.field("description", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("rating", pa.float64()),
            pa.field("rating_count", pa.int64()),
            pa.field("image_url", pa.string()),
            pa.field("product_url", pa.string()),
            pa.field("text_for_embedding", pa.string()),
        ]
    )
    return pa.Table.from_pylist(records, schema=schema)
def _embed_all(repo_root: Path, batch_size: int = 128, max_retries: int = 5, log_every: int = 1000) -> None:
    settings = get_settings()
    data_dir = repo_root / "data"
    products_path = data_dir / "products.parquet"
    if not products_path.exists():
        raise SystemExit(f"Missing products.parquet at {products_path}. Run with --data first.")

    import pyarrow.parquet as pq
    table = pq.read_table(products_path, columns=[
        "id",
        "text_for_embedding",
    ])
    ids_col = table.column("id").to_pylist()
    texts = table.column("text_for_embedding").to_pylist()

    # Filter out empty texts
    filtered_indices: List[int] = []
    filtered_ids: List[str] = []
    filtered_texts: List[str] = []
    for i, (pid, txt) in enumerate(zip(ids_col, texts)):
        if txt and isinstance(txt, str) and txt.strip():
            filtered_indices.append(i)
            filtered_ids.append(pid)
            filtered_texts.append(txt)

    print(f"Embedding {len(filtered_texts)} / {len(texts)} rows (non-empty text)")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    model = settings.model_embed

    embeddings: List[np.ndarray] = []
    for start in range(0, len(filtered_texts), batch_size):
        batch = filtered_texts[start:start + batch_size]
        # Retry with backoff
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
                embeddings.extend(vecs)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = min(2 ** attempt, 30)
                print(f"Embed batch failed (attempt {attempt}/{max_retries}): {e}. Sleeping {sleep_s}s...")
                time.sleep(sleep_s)

        if (start // batch_size) % max(1, (log_every // batch_size)) == 0:
            print(f"Progress: {min(start + batch_size, len(filtered_texts))}/{len(filtered_texts)}")

    if not embeddings:
        raise SystemExit("No embeddings created.")

    mat = np.stack(embeddings, axis=0).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    # Save artifacts
    np.save(data_dir / "embeddings.npy", mat)
    with (data_dir / "ids.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_ids, f)
    stats = {
        "model": model,
        "dim": int(mat.shape[1]),
        "rows": int(mat.shape[0]),
        "created_at": datetime.now(UTC).isoformat() if 'UTC' in globals() else datetime.utcnow().isoformat() + "Z",
    }
    with (data_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Embeddings saved: shape={mat.shape}, dtype={mat.dtype}")


def _build_faiss(repo_root: Path) -> None:
    # Local import to avoid requiring FAISS for non-index flows
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit("FAISS not installed. Prefer conda: 'conda install -c conda-forge faiss-cpu'. Fallback: 'pip install faiss-cpu'.")

    settings = get_settings()
    data_dir = repo_root / "data"
    emb_path = data_dir / "embeddings.npy"
    ids_path = data_dir / "ids.json"
    stats_path = data_dir / "stats.json"
    if not emb_path.exists() or not ids_path.exists() or not stats_path.exists():
        raise SystemExit("Missing embeddings.npy / ids.json / stats.json. Run with --embed first.")

    mat = np.load(emb_path)
    with ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    dim = mat.shape[1]
    if int(stats.get("dim", -1)) != dim:
        raise SystemExit(f"Dimension mismatch: stats.dim={stats.get('dim')} vs emb dim={dim}")
    if len(ids) != mat.shape[0]:
        raise SystemExit(f"Row count mismatch: len(ids)={len(ids)} vs emb rows={mat.shape[0]}")

    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    if index.ntotal != mat.shape[0]:
        raise SystemExit(f"FAISS ntotal mismatch: {index.ntotal} vs {mat.shape[0]}")

    out_path = Path(settings.index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    summary = {"type": "flatip", "dim": int(dim), "size": int(index.ntotal), "path": str(out_path)}
    print(json.dumps(summary))


def run(dataset_path: Path, repo_root: Path) -> int:
    import pyarrow.parquet as pq
    data_dir = repo_root / "data"
    samples_dir = repo_root / "samples"
    _ensure_dir(data_dir)
    _ensure_dir(samples_dir)

    records: List[Dict[str, Any]] = []
    seen_total = 0
    missing_id_examples: List[List[str]] = []
    for raw in _iter_jsonl(dataset_path):
        seen_total += 1
        rec = _normalize_record(raw)
        if not rec.get("id"):
            if isinstance(raw, dict) and len(missing_id_examples) < 3:
                missing_id_examples.append(sorted(list(raw.keys())))
            continue
        records.append(rec)

    table = _to_table(records)

    out_parquet = data_dir / "products.parquet"
    pq.write_table(table, out_parquet)

    # meta.json
    meta = {
        "source": str(dataset_path),
        "row_count": table.num_rows,
        "created_at": datetime.now(UTC).isoformat(),
        "columns": [f.name for f in table.schema],
    }
    with (data_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # mini sample
    mini_rows = min(30, table.num_rows)
    if mini_rows > 0:
        mini = table.slice(0, mini_rows)
        pq.write_table(mini, samples_dir / "mini.parquet")

    print(f"Read records: {seen_total}")
    if missing_id_examples:
        print("Examples of records missing an id-like key (showing up to 3):")
        for keys in missing_id_examples:
            print(f" - keys: {keys}")
    print(f"Rows: {table.num_rows}")
    print(f"Wrote: {out_parquet}")
    print(f"Meta: {(data_dir / 'meta.json')}")
    print(f"Mini: {(samples_dir / 'mini.parquet')}")
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest/Embed/Index pipeline")
    parser.add_argument("--data", dest="data", type=str, default=os.getenv("DATASET_PATH"), help="Path to JSONL dataset")
    parser.add_argument("--embed", action="store_true", help="Compute embeddings for products.parquet and write artifacts")
    parser.add_argument("--build-faiss", action="store_true", help="Build FAISS index from saved embeddings.npy")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.data:
        dataset_path = Path(args.data)
        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        run(dataset_path, repo_root)

    # Embedding and FAISS steps
    if args.embed:
        _embed_all(repo_root)
    if args.build_faiss or args.build_faiss is True:
        _build_faiss(repo_root)


if __name__ == "__main__":
    main()


