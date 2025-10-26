from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.core.faiss_store import FaissStore
from src.core.retriever import set_store
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
        store = FaissStore()
        store.load(
            index_path=settings.index_path,
            ids_path=settings.ids_path,
            stats_path=settings.stats_path,
        )
        set_store(store)

    @app.get("/healthz")
    async def healthz() -> dict[str, str | int]:
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
            return {
                "backend": "faiss",
                "dim": store.dimension,
                "size": store.size,
                "index_type": store.index_type,
            }
        except Exception:
            return {"backend": "faiss", "dim": 0, "size": 0, "index_type": "unknown"}

    return app


app = create_app()


