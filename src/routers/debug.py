from fastapi import APIRouter, HTTPException
from src.telemetry.counters import snapshot

from src.core.retriever import retrieve
from src.core.config import get_settings
from src.models.schemas import RetrieveRequest
from src.core.parser import parse_query, ParserError
from src.core.config import get_settings


router = APIRouter()


@router.get("")
async def ping() -> dict[str, str]:
    return {"message": "debug router ready"}


@router.post("/retrieve")
async def debug_retrieve(req: RetrieveRequest) -> dict:
    try:
        settings = get_settings()
        k = req.k if isinstance(req.k, int) and req.k > 0 else settings.top_k_default
        return retrieve(req.query, k=k)
    except ValueError as e:
        # Likely empty/invalid query after normalization
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/parse")
async def debug_parse(body: dict) -> dict:
    query = (body or {}).get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    try:
        settings = get_settings()
        facets = parse_query(query, model=settings.model_parser, retries=settings.parser_retries)
        return {
            "facets": facets.model_dump(),
            "trace": {
                "model": settings.model_parser,
                "tokens_estimate": 0,
                "normalized": True,
            },
        }
    except ParserError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/stats")
async def stats() -> dict:
    return snapshot()


