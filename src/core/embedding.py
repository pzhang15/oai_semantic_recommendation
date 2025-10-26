from typing import Optional

import numpy as np
from openai import OpenAI

from src.core.config import get_settings
from src.telemetry.timer import span
from src.telemetry.counters import add_usage


def _sanitize(text: str) -> str:
    # Collapse whitespace and trim
    cleaned = " ".join((text or "").split())
    return cleaned


def embed_query(text: str, model: Optional[str] = None) -> np.ndarray:
    settings = get_settings()
    query = _sanitize(text)
    if not query:
        raise ValueError("Query is empty after normalization")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    use_model = model or settings.model_embed
    with span("embed_query") as sp:
        resp = client.embeddings.create(model=use_model, input=[query])
    # usage optional in embeddings
    try:
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        tin = tout = 0
    add_usage("embed_query", use_model, tin, tout, sp.ms)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.reshape(1, -1)
    return (vec / (norm + 1e-12)).astype(np.float32).reshape(1, -1)


