from __future__ import annotations

import os
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    # Python version
    ver = sys.version_info
    if ver < (3, 10):
        _warn(f"Python >=3.10 recommended (found {ver.major}.{ver.minor})")
    else:
        _ok(f"Python {ver.major}.{ver.minor}")

    # FAISS availability (best-effort)
    try:
        import faiss  # type: ignore

        _ok(f"FAISS import ok; threads={getattr(faiss, 'omp_get_max_threads', lambda: '?')()}")
    except Exception as e:
        _warn(f"FAISS not available: {e}")

    # OpenBLAS/MKL threads
    omp = os.getenv("OMP_NUM_THREADS") or "unset"
    mkl = os.getenv("MKL_NUM_THREADS") or "unset"
    obl = os.getenv("OPENBLAS_NUM_THREADS") or "unset"
    _ok(f"BLAS threads OMP={omp} MKL={mkl} OPENBLAS={obl}")

    # Env keys
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        _ok("OPENAI_API_KEY present")
    else:
        _warn("OPENAI_API_KEY missing → LLM features will be disabled; lexical-only retrieval enabled")

    # Data assets
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    idx = os.getenv("INDEX_PATH", str(data / "index.faiss"))
    ids = os.getenv("IDS_PATH", str(data / "ids.json"))
    stats = os.getenv("STATS_PATH", str(data / "stats.json"))
    ok_idx = Path(idx).exists() and Path(ids).exists() and Path(stats).exists()
    if ok_idx:
        _ok("FAISS index artifacts found")
    else:
        _warn("FAISS index artifacts missing; dense retrieval will be disabled")

    # TF-IDF artifacts
    tfidf = Path(os.getenv("TFIDF_PATH", str(data / "tfidf_doc.npz")))
    tfvec = Path(os.getenv("TFIDF_VEC_PATH", str(data / "tfidf_vectorizer.joblib")))
    tfids = Path(os.getenv("TFIDF_IDS_PATH", str(data / "tfidf_ids.json")))
    if tfidf.exists() and tfvec.exists() and tfids.exists():
        _ok("TF-IDF artifacts found")
    else:
        _warn("TF-IDF artifacts missing; will attempt to build during bootstrap if parquet exists")

    # Parquet presence
    pq = Path(os.getenv("PARQUET_PATH", str(data / "products.parquet")))
    if pq.exists():
        _ok(f"Products parquet present: {pq}")
    else:
        _warn(f"Products parquet not found at {pq}")

    print("Preflight complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


