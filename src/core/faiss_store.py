from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class FaissStore:
    def __init__(self) -> None:
        self._index = None
        self._ids: List[str] = []
        self._stats: dict = {}
        self._index_type = "flatip"
        self._meta: dict = {}
        self._signature: str = ""

    @property
    def dimension(self) -> int:
        # Prefer stats; fall back to index dimension if available
        if "dim" in self._stats:
            return int(self._stats.get("dim", 0))
        if self._index is not None:
            return int(getattr(self._index, "d", 0))
        return 0

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def index_type(self) -> str:
        return self._index_type

    def load(self, index_path: str, ids_path: str, stats_path: str, meta_path: str | None = None) -> None:
        try:
            import faiss  # type: ignore
        except Exception:
            raise RuntimeError("FAISS not installed. Install via conda-forge or pip (faiss-cpu).")

        ipath = Path(index_path)
        if not ipath.exists():
            raise FileNotFoundError(f"Index not found: {ipath}")
        with Path(ids_path).open("r", encoding="utf-8") as f:
            self._ids = json.load(f)
        with Path(stats_path).open("r", encoding="utf-8") as f:
            self._stats = json.load(f)

        self._index = faiss.read_index(str(ipath))
        if int(self._stats.get("dim", -1)) != self._index.d:
            raise ValueError(
                f"Dimension mismatch: stats.dim={self._stats.get('dim')} vs index.d={self._index.d}"
            )
        if len(self._ids) != self._index.ntotal:
            raise ValueError(
                f"Row count mismatch: len(ids)={len(self._ids)} vs index.ntotal={self._index.ntotal}"
            )
        # optional meta
        if meta_path:
            p = Path(meta_path)
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        self._meta = json.load(f)
                    import hashlib
                    self._signature = hashlib.sha1(json.dumps(self._meta, sort_keys=True).encode("utf-8")).hexdigest()
                    self._index_type = str(self._meta.get("mode") or self._index_type)
                except Exception:
                    self._meta = {}
                    self._signature = ""

    def search(self, query_vec: np.ndarray, k: int, *, hnsw_ef: int | None = None) -> Tuple[List[int], List[float]]:
        if self._index is None:
            raise RuntimeError("Index not loaded.")
        # Ensure shape (1, d) and unit norm
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != self._index.d:
            raise ValueError(f"Query dim {query_vec.shape[1]} != index dim {self._index.d}")
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = (query_vec / norms).astype(np.float32)
        try:
            import faiss  # type: ignore
        except Exception:
            faiss = None  # type: ignore
        if hnsw_ef and faiss is not None and hasattr(faiss, "SearchParametersHNSW"):
            params = faiss.SearchParametersHNSW()
            params.efSearch = int(hnsw_ef)
            scores, indices = self._index.search(q, k, params)
        else:
            scores, indices = self._index.search(q, k)
        idxs = [int(i) for i in indices[0]]
        scs = [float(s) for s in scores[0]]
        return idxs, scs

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def signature(self) -> str:
        return self._signature


