from __future__ import annotations

from functools import lru_cache
from openai import OpenAI
import logging

from src.core.config import get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    s = get_settings()
    base_url = s.openai_base_url or None
    # Guard against invalid values like "api.openai.com/v1" (missing scheme) or empty strings
    if base_url and not base_url.strip().lower().startswith(("http://", "https://")):
        logging.warning("OPENAI_BASE_URL missing scheme, ignoring value: %s", base_url)
        base_url = None
    return OpenAI(api_key=s.openai_api_key, base_url=base_url)


