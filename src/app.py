from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import os

from src.core.config import get_settings
from src.core.faiss_store import FaissStore
from src.core.retriever import set_store
from src.core import result_cache as result_cache
from src.core import judge_cache as judge_cache
from src.routers.search import router as search_router
from src.routers.outfit import router as outfit_router
from src.routers.debug import router as debug_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="semantic-rec", version="0.1.0")

    # CORS
    allow_origins = settings.allowed_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers (no business logic yet)
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(outfit_router, prefix="/outfit", tags=["outfit"])
    app.include_router(debug_router, prefix="/debug", tags=["debug"])

    @app.on_event("startup")
    async def _load_faiss() -> None:
        settings = get_settings()
        try:
            store = FaissStore()
            store.load(
                index_path=settings.index_path,
                ids_path=settings.ids_path,
                stats_path=settings.stats_path,
                meta_path=getattr(settings, "index_meta_path", None),
            )
            set_store(store)
        except Exception as e:
            # Degrade gracefully when FAISS artifacts are missing; lexical-only will still work
            try:
                # simple stderr notice without adding a logger dependency
                print(f"[WARN] FAISS index not loaded at startup: {e}")
            except Exception:
                pass
        # Configure FAISS and BLAS threading for performance
        try:
            import faiss  # type: ignore
            threads = max(1, os.cpu_count() or 8)
            faiss.omp_set_num_threads(threads)
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, threads - 1)))
            os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])  # type: ignore[index]
            os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])  # type: ignore[index]
        except Exception:
            pass
        # Configure result cache
        if settings.result_cache_enable:
            result_cache.configure(max_size=settings.result_cache_size, ttl_secs=settings.result_cache_ttl_secs)
        # Configure judge request-level cache (LRU+TTL)
        judge_cache.init(size=settings.judge_cache_size, ttl_secs=settings.judge_cache_ttl_secs)
        # Warm lexical artifacts (TF-IDF) so first request doesn't pay load cost
        try:
            from src.core.lexical import ready as lexical_ready
            lexical_ready()
        except Exception:
            pass

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        # Attempt to read FAISS store info
        try:
            settings = get_settings()
            # The set_store call configured a global store; we reload a lightweight store to introspect
            store = FaissStore()
            store.load(
                index_path=settings.index_path,
                ids_path=settings.ids_path,
                stats_path=settings.stats_path,
            )
            resp = {
                "backend": "faiss",
                "dim": store.dimension,
                "size": store.size,
                "index_type": store.index_type,
                "llm_enabled": bool(getattr(settings, "llm_enabled", False)),
            }
            # lexical info if present
            try:
                from src.core.lexical import ready as lexical_ready
                lx = lexical_ready()
                resp["lexical"] = lx
            except Exception:
                resp["lexical"] = {"ready": False}
            return resp
        except Exception:
            s = get_settings()
            return {
                "backend": "faiss",
                "dim": 0,
                "size": 0,
                "index_type": "unknown",
                "llm_enabled": bool(getattr(s, "llm_enabled", False)),
                "lexical": {"ready": False},
            }

    return app


app = create_app()


