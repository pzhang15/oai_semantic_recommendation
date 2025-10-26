from __future__ import annotations

from typing import Dict


PRICING_USD_PER_1K: Dict[str, Dict[str, float]] = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING_USD_PER_1K.get(model) or PRICING_USD_PER_1K.get(model.lower())
    if not p:
        return 0.0
    return (prompt_tokens / 1000.0) * p.get("input", 0.0) + (completion_tokens / 1000.0) * p.get("output", 0.0)


