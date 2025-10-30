from __future__ import annotations

import os
from pathlib import Path


def _log(msg: str) -> None:
    print(msg)


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    index_path = os.getenv("INDEX_PATH", str(data / "index.faiss"))
    ids_path = os.getenv("IDS_PATH", str(data / "ids.json"))
    stats_path = os.getenv("STATS_PATH", str(data / "stats.json"))

    tfidf_path = os.getenv("TFIDF_PATH", str(data / "tfidf_doc.npz"))
    tfidf_vec_path = os.getenv("TFIDF_VEC_PATH", str(data / "tfidf_vectorizer.joblib"))
    tfidf_ids_path = os.getenv("TFIDF_IDS_PATH", str(data / "tfidf_ids.json"))

    parquet_path = os.getenv("PARQUET_PATH", str(data / "products.parquet"))

    # 1) Ensure lexical artifacts exist (can be built without LLM)
    if not (_exists(tfidf_path) and _exists(tfidf_vec_path) and _exists(tfidf_ids_path)):
        if _exists(parquet_path):
            _log("Building TF-IDF artifacts for lexical retrieval...")
            try:
                from src.core.lexical_build import build_lexical

                build_lexical(root)
            except Exception as e:
                _log(f"[WARN] Failed to build TF-IDF artifacts: {e}")
        else:
            _log("[WARN] Parquet not found; skipping TF-IDF build")

    # 2) If FAISS artifacts are missing, we do not attempt to build embeddings (requires LLM)
    if not (_exists(index_path) and _exists(ids_path) and _exists(stats_path)):
        _log("[WARN] FAISS index artifacts missing. Dense retrieval will be disabled.")

    _log("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


