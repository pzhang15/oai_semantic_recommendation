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
import re

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
        s = value.strip()
        # handle ranges like "19.99-29.99" -> choose lower bound
        if any(ch in s for ch in ["-", "–", "—", " to "]):
            parts = [p for p in re.split(r"\s*(?:-|–|—|to)\s*", s) if p]
            for p in parts:
                try:
                    return float(p.replace("$", "").replace(",", ""))
                except Exception:
                    continue
            return None
        s = s.replace("$", "").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(value, dict):
        # Common dict shapes: {amount, currency} or {value, currency}
        for k in ["amount", "value", "price", "usd", "unit_amount"]:
            v = value.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.replace("$", "").replace(",", "").strip())
                except Exception:
                    continue
        # Fallback: scan all values for numeric-like
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.replace("$", "").replace(",", "").strip())
                except Exception:
                    pass
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
def _infer_brand_from_title(title: str | None) -> Optional[str]:
    if not title:
        return None
    t = str(title)
    for marker in [" Men's", " Men's ", " men'", " Women's", " women's", " - ", " – ", ": "]:
        idx = t.find(marker)
        if idx > 0:
            cand = t[:idx].strip()
            if 2 <= len(cand) <= 80:
                return cand
    parts = t.split()
    if parts and parts[0][0:1].isupper():
        head = parts[0]
        if len(parts) > 1 and parts[1][0:1].isupper():
            head += " " + parts[1]
        return head
    return None


def _extract_brand(raw: Dict[str, Any], title: str) -> Optional[str]:
    # Tier 1: direct keys
    brand = _to_string(_pick_first("brand", "brand_name", "manufacturer", "by", from_obj=raw))
    if brand:
        return brand
    # Tier 2: details dict
    det = raw.get("details")
    if isinstance(det, dict):
        for k in ["Brand", "brand", "Manufacturer", "maker", "label", "By"]:
            v = det.get(k)
            if v not in (None, ""):
                s = _to_string(v)
                if s:
                    return s
    # Tier 3: title inference
    inferred = _infer_brand_from_title(title)
    if inferred:
        return inferred
    return None


def _extract_product_url(raw: Dict[str, Any], asin: str) -> Optional[str]:
    for k in [
        "product_url",
        "url",
        "detail_page_url",
        "productURL",
        "product_link",
        "link",
        "detail_url",
        "canonical_url",
        "productUrl",
    ]:
        v = raw.get(k)
        if v not in (None, ""):
            return _to_string(v)
    if asin:
        return f"https://www.amazon.com/dp/{asin}"
    return None


def _parse_int_with_suffix(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    m = re.search(r"([\d,.]+)\s*([kKmM]?)", t)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        val = float(num)
    except Exception:
        return None
    suf = m.group(2)
    if suf in ("k", "K"):
        val *= 1_000
    elif suf in ("m", "M"):
        val *= 1_000_000
    return int(val)


def _extract_rating_count(raw: Dict[str, Any]) -> Optional[int]:
    rc = _pick_first(
        "rating_count",
        "review_count",
        "vote",
        "ratings_count",
        "ratings_total",
        "total_ratings",
        from_obj=raw,
    )
    if isinstance(rc, (int, float)):
        return int(rc)
    if isinstance(rc, str):
        val = _parse_int_with_suffix(rc)
        if val is not None:
            return val
    # details fallback
    det = raw.get("details")
    if isinstance(det, dict):
        for k in ["ratings", "reviews", "review_count", "ratings_total", "votes"]:
            v = det.get(k)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                vv = _parse_int_with_suffix(v)
                if vv is not None:
                    return vv
    return None


def _extract_description(raw: Dict[str, Any], features: List[str]) -> tuple[str, List[str]]:
    desc_str: str = ""
    desc_list: List[str] = []
    dv = _pick_first("description", from_obj=raw)
    if isinstance(dv, list):
        desc_list = _flatten_list(dv)
        desc_str = " ".join(desc_list)
    elif isinstance(dv, str):
        desc_str = _to_string(dv)
    # explicit description_list key
    dl = raw.get("description_list")
    if isinstance(dl, list) and not desc_list:
        desc_list = _flatten_list(dl)
    if not desc_str and desc_list:
        desc_str = " ".join(desc_list)
    # fallback to features when empty
    if not desc_str and features:
        desc_list = desc_list or list(features)
        desc_str = " ".join(features)
    return desc_str, desc_list



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
    main_category = _to_string(_pick_first("main_category", "domain", from_obj=raw))
    brand = _extract_brand(raw, title)

    categories = _flatten_list(_pick_first("categories", from_obj=raw))
    features = _flatten_list(_pick_first("feature", "features", from_obj=raw))
    description, description_list = _extract_description(raw, features)

    price = _parse_price(_pick_first("price", "current_price", "sale_price", "list_price", from_obj=raw))

    rating = None
    rating_src = _pick_first("rating", "average_rating", "overall", from_obj=raw)
    if isinstance(rating_src, (int, float)):
        rating = float(rating_src)
    elif isinstance(rating_src, str):
        try:
            rating = float(rating_src)
        except ValueError:
            rating = None

    rating_count = _extract_rating_count(raw)

    def _extract_image_url(obj: Dict[str, Any]) -> Optional[str]:
        # Common direct keys
        for k in [
            "imageURLHighRes",
            "imageURL",
            "imUrl",
            "image_url",
            "image"
        ]:
            if k in obj and obj[k] not in (None, ""):
                v = obj[k]
                if isinstance(v, list) and v:
                    s = _to_string(v[0])
                    if s:
                        return s
                elif isinstance(v, dict):
                    for kk in ["large", "medium", "small", "url", "link"]:
                        if kk in v and v[kk]:
                            s = _to_string(v[kk])
                            if s:
                                return s
                else:
                    s = _to_string(v)
                    if s:
                        return s
        # Alternate list-like keys
        for k in ["images", "imageUrls", "image_urls", "images_highres", "pictures"]:
            v = obj.get(k)
            if isinstance(v, list) and v:
                # pick first non-empty string or dict url
                for item in v:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        for kk in ["url", "link", "large", "medium", "small"]:
                            if kk in item and item[kk]:
                                s = _to_string(item[kk])
                                if s:
                                    return s
        return None

    image_url = _extract_image_url(raw)

    product_url = _extract_product_url(raw, asin)

    # Images: collect list of URLs if present
    images: List[str] = []
    for key in ["imageURLHighRes", "imageURL", "images", "image_urls", "imageUrls", "images_highres", "pictures"]:
        v = raw.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    images.append(item.strip())
                elif isinstance(item, dict):
                    for kk in ["hi_res", "large", "medium", "small", "url", "link"]:
                        if kk in item and item[kk]:
                            s = _to_string(item[kk])
                            if s:
                                images.append(s)
        elif isinstance(v, str) and v.strip():
            images.append(v.strip())
        elif isinstance(v, dict):
            for kk in ["hi_res", "large", "medium", "small", "url", "link"]:
                if kk in v and v[kk]:
                    s = _to_string(v[kk])
                    if s:
                        images.append(s)
    # ensure primary image_url if not set
    if not image_url and images:
        image_url = images[0]

    # Videos: list of URLs if available
    videos: List[str] = []
    vsrc = raw.get("videos") or raw.get("video")
    if isinstance(vsrc, list):
        for item in vsrc:
            if isinstance(item, str) and item.strip():
                videos.append(item.strip())
            elif isinstance(item, dict):
                for kk in ["url", "link", "src"]:
                    if kk in item and item[kk]:
                        s = _to_string(item[kk])
                        if s:
                            videos.append(s)
    elif isinstance(vsrc, dict):
        for kk in ["url", "link", "src"]:
            if kk in vsrc and vsrc[kk]:
                s = _to_string(vsrc[kk])
                if s:
                    videos.append(s)

    store = _to_string(_pick_first("store", "store_name", "seller", from_obj=raw))
    parent_asin = _to_string(_pick_first("parent_asin", "parentAsin", from_obj=raw))
    details_obj = raw.get("details") if isinstance(raw, dict) else None
    details = json.dumps(details_obj, ensure_ascii=False) if isinstance(details_obj, dict) else None
    bought_together_list = raw.get("bought_together") or raw.get("also_bought") or []
    if not isinstance(bought_together_list, list):
        bought_together = []
    else:
        bought_together = [str(x) for x in bought_together_list if x is not None]

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
        "parent_asin": parent_asin or None,
        "main_category": main_category or None,
        "title": title or None,
        "brand": brand or None,
        "categories": categories,
        "features": features,
        "description": description or None,
        "description_list": description_list,
        "price": price,
        "rating": rating,
        "average_rating": rating,
        "rating_count": rating_count,
        "rating_number": rating_count,
        "image_url": image_url or None,
        "product_url": product_url or None,
        "images": images,
        "videos": videos,
        "store": store or None,
        "details": details,
        "bought_together": bought_together,
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
            pa.field("parent_asin", pa.string()),
            pa.field("main_category", pa.string()),
            pa.field("title", pa.string()),
            pa.field("brand", pa.string()),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("features", pa.list_(pa.string())),
            pa.field("description", pa.string()),
            pa.field("description_list", pa.list_(pa.string())),
            pa.field("price", pa.float64()),
            pa.field("rating", pa.float64()),
            pa.field("average_rating", pa.float64()),
            pa.field("rating_count", pa.int64()),
            pa.field("rating_number", pa.int64()),
            pa.field("image_url", pa.string()),
            pa.field("product_url", pa.string()),
            pa.field("images", pa.list_(pa.string())),
            pa.field("videos", pa.list_(pa.string())),
            pa.field("store", pa.string()),
            pa.field("details", pa.string()),
            pa.field("bought_together", pa.list_(pa.string())),
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
    parser.add_argument("--build-lexical", action="store_true", help="Build TF-IDF lexical index artifacts")
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
    if args.build_lexical:
        from src.core.lexical_build import build_lexical
        out = build_lexical(repo_root)
        print(json.dumps(out))


if __name__ == "__main__":
    main()


