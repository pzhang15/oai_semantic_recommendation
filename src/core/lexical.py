from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import load
from scipy import sparse

from src.core.config import get_settings


@lru_cache
def _load_artifacts() -> Tuple[Any, sparse.csr_matrix, List[str]]:
    settings = get_settings()
    vec_path = Path(settings.tfidf_vec_path)
    doc_path = Path(settings.tfidf_path)
    ids_path = Path(settings.tfidf_ids_path)
    if not (vec_path.exists() and doc_path.exists() and ids_path.exists()):
        raise FileNotFoundError("TF-IDF artifacts not found")
    vectorizer = load(vec_path)
    X_docs = sparse.load_npz(doc_path).tocsr()
    with ids_path.open("r", encoding="utf-8") as f:
        ids: List[str] = [str(x) for x in json.load(f)]
    return vectorizer, X_docs, ids


def ready() -> Dict[str, Any]:
    try:
        vectorizer, X_docs, ids = _load_artifacts()
        vocab_size = len(getattr(vectorizer, "vocabulary_", {}))
        return {"ready": True, "rows": int(X_docs.shape[0]), "vocab": int(vocab_size), "path": get_settings().tfidf_path}
    except Exception:
        return {"ready": False}


def _compose_query_text(query_text: str, facets: Any | None) -> str:
    txt = (query_text or "").strip()
    if not facets:
        return txt
    tokens: List[str] = []
    try:
        # append a few facet tokens lightly
        for arr in [getattr(facets, "colors", None) or [], getattr(facets, "materials", None) or [], getattr(facets, "categories_include", None) or []]:
            for t in arr:
                s = str(t).strip().lower()
                if s and s not in tokens:
                    tokens.append(s)
        if tokens:
            extra = " ".join(tokens[:30])
            txt = f"{txt} {extra}" if txt else extra
    except Exception:
        pass
    return txt


def search_lexical(query_text: str, k: int, facets: Any | None = None, return_timings: bool = False):
    import time as _t
    t0 = _t.perf_counter()
    vectorizer, X_docs, ids = _load_artifacts()
    q = _compose_query_text(query_text, facets)
    t1 = _t.perf_counter()
    Xq = vectorizer.transform([q])
    t2 = _t.perf_counter()
    sims = (X_docs @ Xq.T).toarray().ravel()
    t3 = _t.perf_counter()
    out: List[Dict[str, Any]] = []
    if sims.size != 0:
        nz = sims.nonzero()[0]
        if nz.size != 0:
            cand_idx = nz
            cand_scores = sims[nz]
            k_eff = int(min(max(1, k), cand_scores.size))
            part = np.argpartition(cand_scores, -k_eff)[-k_eff:]
            subset_idx = cand_idx[part]
            subset_scores = cand_scores[part]
            order = np.lexsort((subset_idx.astype(str), -subset_scores))
            top_idx = subset_idx[order]
            for rank, i in enumerate(top_idx, start=1):
                out.append({"id": ids[int(i)], "score": float(sims[int(i)]), "rank": rank})
    t4 = _t.perf_counter()
    if return_timings:
        timings = {
            "lex_compose_ms": round((t1 - t0) * 1000.0, 2),
            "lex_transform_ms": round((t2 - t1) * 1000.0, 2),
            "lex_matvec_ms": round((t3 - t2) * 1000.0, 2),
            "lex_topk_ms": round((t4 - t3) * 1000.0, 2),
        }
        return out, timings
    return out


