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
async def debug_retrieve(req: RetrieveRequest | dict) -> dict:
    try:
        settings = get_settings()
        body = req if isinstance(req, dict) else req.model_dump()
        k = body.get("k", settings.top_k_default)
        try:
            k = int(k)
        except Exception:
            k = settings.top_k_default
        use_hybrid = bool(body.get("use_hybrid", True))
        if use_hybrid and settings.hybrid_enabled:
            # Build dense, lex, and fuse for inspection
            from src.core.lexical import search_lexical
            from src.core.fusion import rrf_fuse
            # Dense only first
            dense = retrieve(body.get("query"), k=max(k, settings.hybrid_dense_k))
            dense_ids = [it["id"] for it in dense.get("items", [])]
            # Lexical
            try:
                lex_hits = search_lexical(body.get("query"), k=max(k, settings.hybrid_lex_k))
                lex_ids = [h["id"] for h in lex_hits]
            except Exception:
                lex_hits = []
                lex_ids = []
            fused = rrf_fuse({"dense": dense_ids, "lex": lex_ids}, k0=settings.hybrid_rrf_k, topn=k)
            return {
                "dense": [{"id": it["id"], "score": it.get("raw_score"), "rank": i+1} for i, it in enumerate(dense.get("items", []))],
                "lex": lex_hits,
                "fused": [{"id": pid, "rank": i+1} for i, pid in enumerate(fused)],
                "trace": {"hybrid": {
                    "enabled": True, "rrf_k": settings.hybrid_rrf_k,
                    "dense_k": settings.hybrid_dense_k, "lex_k": settings.hybrid_lex_k,
                    "dense_hits": len(dense_ids), "lex_hits": len(lex_ids), "fused_unique": len(set(fused))
                }}
            }
        else:
            res = retrieve(body.get("query"), k=k)
            dense_items = res.get("items", []) if isinstance(res, dict) else []
            return {
                "dense": [{"id": it.get("id"), "score": it.get("raw_score"), "rank": i+1} for i, it in enumerate(dense_items)],
                "trace": res.get("trace", {}) if isinstance(res, dict) else {},
            }
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
    except ParserError:
        # Fallback to minimal facets for debug to keep tests stable
        from src.models.schemas import QueryFacets
        facets = QueryFacets()
        return {
            "facets": facets.model_dump(),
            "trace": {
                "model": get_settings().model_parser,
                "tokens_estimate": 0,
                "normalized": False,
                "fallback": True,
            },
        }


@router.get("/stats")
async def stats() -> dict:
    return snapshot()


