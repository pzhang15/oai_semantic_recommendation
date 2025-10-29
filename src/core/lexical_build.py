from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from scipy import sparse

from src.core.config import get_settings


def _join_list(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        # dedupe, keep order
        seen = set()
        out: List[str] = []
        for v in values:
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return " ".join(out)
    return str(values)


def _compose_doc(row: pd.Series, fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        val = row.get(f)
        if f in ("categories", "features"):
            text = _join_list(val)
        else:
            text = str(val) if val is not None else ""
        if text:
            if f == "title":
                parts.append(f"title: {text}")
            elif f == "brand":
                parts.append(f"brand: {text}")
            elif f == "categories":
                parts.append(f"cats: {text}")
            elif f == "features":
                parts.append(f"feats: {text}")
    return " \n".join(parts)


def build_lexical(repo_root: Path | None = None) -> dict:
    settings = get_settings()
    root = repo_root or Path(__file__).resolve().parents[2]
    products = Path(settings.parquet_path)
    if not products.exists():
        raise SystemExit(f"Missing products parquet at {products}")

    cols = ["id", *settings.lexical_fields]
    df = pd.read_parquet(products, columns=[c for c in cols if c in pd.read_parquet(products, columns=None).columns])
    if "id" not in df.columns:
        raise SystemExit("products.parquet missing 'id' column")

    docs = df.apply(lambda r: _compose_doc(r, settings.lexical_fields), axis=1).tolist()
    ids = df["id"].astype(str).tolist()

    vec = TfidfVectorizer(
        lowercase=True,
        min_df=settings.lexical_min_df,
        max_df=settings.lexical_max_df,
        ngram_range=settings.lexical_ngrams,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\']+\b",
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,  # speed + smaller memory
    )
    X = vec.fit_transform(docs)

    # Persist artifacts
    Path(settings.tfidf_vec_path).parent.mkdir(parents=True, exist_ok=True)
    dump(vec, settings.tfidf_vec_path)
    sparse.save_npz(settings.tfidf_path, X.tocsr())
    with Path(settings.tfidf_ids_path).open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    meta = {
        "rows": int(X.shape[0]),
        "vocab_size": int(len(vec.vocabulary_)),
        "min_df": settings.lexical_min_df,
        "max_df": settings.lexical_max_df,
        "ngrams": list(settings.lexical_ngrams),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with Path(settings.tfidf_meta_path).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"lexical": {"rows": meta["rows"], "vocab": meta["vocab_size"], "path": settings.tfidf_path}}


