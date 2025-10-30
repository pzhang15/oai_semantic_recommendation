import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_head(arr: np.ndarray, n: int = 5) -> str:
    import hashlib
    head = arr[:n].astype("float32").tobytes()
    return hashlib.sha256(head).hexdigest()


def _load_emb_ids(emb_path: Path, ids_path: Path) -> tuple[np.ndarray, list[str]]:
    emb = np.load(emb_path)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list):
        raise ValueError("ids.json must be a list")
    if emb.shape[0] != len(ids):
        raise ValueError(f"rows mismatch: emb={emb.shape[0]} ids={len(ids)}")
    # L2-normalize (safety)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = (emb / norms).astype(np.float32)
    return emb, [str(x) for x in ids]


def build_index(args: argparse.Namespace) -> None:
    try:
        import faiss  # type: ignore
    except Exception as e:
        print("FAISS not installed:", e, file=sys.stderr)
        sys.exit(2)

    emb, ids = _load_emb_ids(Path(args.emb), Path(args.ids))
    dim_in = emb.shape[1]
    mode = args.mode.lower()
    meta: dict[str, Any] = {
        "mode": mode,
        "dim_in": dim_in,
        "dim_out": dim_in,
        "count": int(emb.shape[0]),
        "emb_sha256_head": _sha256_head(emb),
    }
    if args.threads:
        try:
            faiss.omp_set_num_threads(int(args.threads))
        except Exception:
            pass

    if mode == "flat":
        index = faiss.IndexFlatIP(dim_in)
        index.add(emb)
    elif mode == "hnsw_1536":
        index = faiss.IndexHNSWFlat(dim_in, int(args.hnsw_m))
        index.hnsw.efConstruction = int(args.efc)
        index.add(emb)
    elif mode in {"pca_flat", "hnsw_pca"}:
        dim_out = int(args.dim_out or 256)
        meta["dim_out"] = dim_out
        pca = faiss.PCAMatrix(dim_in, dim_out)
        # train on a sample up to 100k
        n_train = min(emb.shape[0], 100_000)
        pca.train(emb[:n_train])
        if mode == "pca_flat":
            base = faiss.IndexFlatIP(dim_out)
        else:
            base = faiss.IndexHNSWFlat(dim_out, int(args.hnsw_m))
            base.hnsw.efConstruction = int(args.efc)
        index = faiss.IndexPreTransform(pca, base)
        index.add(emb)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    faiss.write_index(index, args.out)
    meta_path = Path(args.meta)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out} and {args.meta}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ANN indexes (flat/hnsw/pca)")
    ap.add_argument("--mode", required=True, help="flat|pca_flat|hnsw_1536|hnsw_pca")
    ap.add_argument("--emb", required=True, help="Path to embeddings.npy")
    ap.add_argument("--ids", required=True, help="Path to ids.json")
    ap.add_argument("--out", required=True, help="Output index path (.faiss)")
    ap.add_argument("--meta", required=True, help="Output meta path (.meta.json)")
    ap.add_argument("--dim_out", type=int, default=256)
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 8))
    args = ap.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()


