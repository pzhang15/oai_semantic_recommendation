from __future__ import annotations

from functools import lru_cache
from openai import OpenAI

from src.core.config import get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)


