from __future__ import annotations

from typing import Any, List, Set, Tuple, Dict

import numpy as np

from src.core.retriever import get_store


def _tokenize_title(t: str) -> Set[str]:
    return {w for w in (t or "").lower().replace("/", " ").replace("-", " ").split() if w and w.isalpha()}


def _title_similarity(a: str, b: str) -> float:
    ta = _tokenize_title(a)
    tb = _tokenize_title(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(union) if union else 0.0


def _build_title_stem(t: str) -> str:
    tokens = sorted(_tokenize_title(t))
    return " ".join(tokens[:10])


def _reconstruct_vector(vec_idx: int) -> np.ndarray | None:
    store = get_store()
    if store is None:
        return None
    try:
        import faiss  # type: ignore
        if hasattr(store, "_index") and store._index is not None:  # type: ignore[attr-defined]
            v = np.zeros((store._index.d,), dtype=np.float32)  # type: ignore[attr-defined]
            store._index.reconstruct(vec_idx, v)  # type: ignore[attr-defined]
            return v
    except Exception:
        return None
    return None


def _similarity(a: dict[str, Any], b: dict[str, Any], *, mode: str, title_threshold: float, vec_cache: Dict[int, np.ndarray]) -> Tuple[float, bool]:
    used_vec = False
    if mode in {"vector_only", "vector_then_title"}:
        ai = a.get("vector_index")
        bi = b.get("vector_index")
        if isinstance(ai, int) and isinstance(bi, int) and ai >= 0 and bi >= 0:
            va = vec_cache.get(ai)
            if va is None:
                if "vector" in a and isinstance(a["vector"], (list, tuple, np.ndarray)):
                    va = np.asarray(a["vector"], dtype=np.float32)
                else:
                    va = _reconstruct_vector(ai)
                if va is not None:
                    vec_cache[ai] = va
            vb = vec_cache.get(bi)
            if vb is None:
                if "vector" in b and isinstance(b["vector"], (list, tuple, np.ndarray)):
                    vb = np.asarray(b["vector"], dtype=np.float32)
                else:
                    vb = _reconstruct_vector(bi)
                if vb is not None:
                    vec_cache[bi] = vb
            if va is not None and vb is not None:
                # vectors are unit-normalized; cosine = dot
                sim = float(np.dot(va.astype(np.float32), vb.astype(np.float32)))
                used_vec = True
                return sim, used_vec
    if mode in {"title_only", "vector_then_title"}:
        sim = _title_similarity(a.get("title") or "", b.get("title") or "")
        return sim, used_vec
    return 0.0, used_vec


def mmr_select(
    items: List[dict[str, Any]],
    *,
    lambda_mult: float = 0.7,
    final_k: int = 12,
    mode: str = "vector_then_title",
    title_threshold: float = 0.85,
    dedup_keys: List[str] | None = None,
    variant_keys: List[str] | None = None,
) -> Tuple[List[dict[str, Any]], Dict[str, Any]]:
    if not items:
        return [], {"lambda": lambda_mult, "mode": mode, "final": 0, "dedup_dropped": 0, "variant_capped": 0, "used_vector_similarity": False}

    dedup_keys = dedup_keys or ["id"]
    variant_keys = variant_keys or ["parent_asin", "title_stem", "brand"]

    # Precompute rel normalization and stems
    rels = [float(it.get("score") or it.get("raw_score") or 0.0) for it in items]
    rmax = max(rels) if rels else 0.0
    def rel_norm(it: dict[str, Any]) -> float:
        r = float(it.get("score") or it.get("raw_score") or 0.0)
        return (r / rmax) if rmax > 0 else 0.0

    for it in items:
        if not it.get("title_stem"):
            it["title_stem"] = _build_title_stem(it.get("title") or "")

    # Exact dedup
    seen_keys: Set[Tuple[Any, ...]] = set()
    dedup_dropped = 0
    filtered: List[dict[str, Any]] = []
    for it in items:
        key = tuple(it.get(k) for k in dedup_keys)
        if key in seen_keys:
            dedup_dropped += 1
            continue
        seen_keys.add(key)
        filtered.append(it)

    # Greedy MMR with variant capping
    k = max(1, min(final_k, len(filtered)))
    remaining = list(filtered)
    remaining.sort(key=lambda x: (rel_norm(x), float(x.get("score") or x.get("raw_score") or 0.0), -(x.get("vector_index") or 0)), reverse=True)
    selected: List[dict[str, Any]] = []
    vec_cache: Dict[int, np.ndarray] = {}
    used_vec_any = False
    variant_capped = 0
    variant_counts: Dict[Tuple[Any, ...], int] = {}

    def variant_key(it: dict[str, Any]) -> Tuple[Any, ...]:
        vals: List[Any] = []
        for k in variant_keys or []:
            vals.append(it.get(k))
        return tuple(vals)

    def allowed_by_variant(it: dict[str, Any]) -> bool:
        key = variant_key(it)
        cnt = variant_counts.get(key, 0)
        return cnt < 2

    while remaining and len(selected) < k:
        best_idx = 0
        best_val = -1e9
        best_used_vec = False
        for i, cand in enumerate(remaining):
            sim_max = 0.0
            used_vec_local = False
            if selected:
                for s in selected:
                    sim, used_vec = _similarity(cand, s, mode=mode, title_threshold=title_threshold, vec_cache=vec_cache)
                    if used_vec:
                        used_vec_local = True
                    if sim > sim_max:
                        sim_max = sim
            rel = rel_norm(cand)
            val = lambda_mult * rel - (1.0 - lambda_mult) * sim_max
            # stable tie-break: prefer higher rel, then lower vector_index
            tie = (val, rel, -int(cand.get("vector_index") or 0))
            if val > best_val or (abs(val - best_val) < 1e-12 and tie > (best_val, 0.0, 0)):
                best_val = val
                best_idx = i
                best_used_vec = used_vec_local
        cand = remaining.pop(best_idx)
        if not allowed_by_variant(cand):
            variant_capped += 1
            continue
        used_vec_any = used_vec_any or best_used_vec
        selected.append(cand)
        key = variant_key(cand)
        variant_counts[key] = variant_counts.get(key, 0) + 1

    trace = {
        "lambda": lambda_mult,
        "mode": mode,
        "final": len(selected),
        "dedup_dropped": dedup_dropped,
        "variant_capped": variant_capped,
        "used_vector_similarity": used_vec_any,
    }
    return selected, trace


