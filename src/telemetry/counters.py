from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.telemetry.costs import estimate_cost


START_TIME = time.time()
WINDOW_SECS = 3600.0


@dataclass
class ModelUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd_est: float = 0.0


@dataclass
class ComponentUsage:
    calls: int = 0
    ms_total: float = 0.0
    events: List[Tuple[float, float]] = field(default_factory=list)  # (ts, ms)


_lock = threading.Lock()
_models: Dict[str, ModelUsage] = {}
_components: Dict[str, ComponentUsage] = {}
# Judge gate counters
_judge_skipped = 0
_judge_cheap = 0
_judge_full = 0
_judge_escalations = 0
_judge_cache_hits = 0
_judge_cache_misses = 0
_judge_cost_saved_usd_est = 0.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def add_usage(component: str, model: str | None, tokens_in: int, tokens_out: int, ms: float) -> None:
    with _lock:
        comp = _components.setdefault(component, ComponentUsage())
        comp.calls += 1
        comp.ms_total += float(ms)
        comp.events.append((time.time(), float(ms)))
        # trim window
        cutoff = time.time() - WINDOW_SECS
        comp.events = [e for e in comp.events if e[0] >= cutoff]

        if model:
            mu = _models.setdefault(model, ModelUsage())
            mu.calls += 1
            mu.tokens_in += int(tokens_in or 0)
            mu.tokens_out += int(tokens_out or 0)
            mu.cost_usd_est += estimate_cost(model, int(tokens_in or 0), int(tokens_out or 0))


def add_judge_gate_event(*, mode: str, cache_hit: bool, escalated: bool, cost_saved_usd: float) -> None:
    global _judge_skipped, _judge_cheap, _judge_full, _judge_escalations, _judge_cache_hits, _judge_cache_misses, _judge_cost_saved_usd_est
    with _lock:
        if mode == "skip":
            _judge_skipped += 1
        elif mode == "cheap":
            _judge_cheap += 1
        elif mode == "full":
            _judge_full += 1
        if escalated:
            _judge_escalations += 1
        if cache_hit:
            _judge_cache_hits += 1
        else:
            _judge_cache_misses += 1
        _judge_cost_saved_usd_est += float(cost_saved_usd or 0.0)


def estimate_model_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    return float(estimate_cost(model, int(tokens_in or 0), int(tokens_out or 0)))


def snapshot() -> Dict[str, Any]:
    with _lock:
        models = {m: {
            "calls": mu.calls,
            "tokens_in": mu.tokens_in,
            "tokens_out": mu.tokens_out,
            "cost_usd_est": round(mu.cost_usd_est, 4),
        } for m, mu in _models.items()}

        comps: Dict[str, Any] = {}
        for name, cu in _components.items():
            vals = [ms for (_, ms) in cu.events]
            comps[name] = {
                "calls": cu.calls,
                "ms_total": round(cu.ms_total, 2),
                "p50_ms": round(_percentile(vals, 50), 2),
                "p95_ms": round(_percentile(vals, 95), 2),
            }

        return {
            "uptime_secs": int(time.time() - START_TIME),
            "models": models,
            "components": comps,
            "judge_gate": {
                "skipped": _judge_skipped,
                "cheap": _judge_cheap,
                "full": _judge_full,
                "escalations": _judge_escalations,
                "cache_hits": _judge_cache_hits,
                "cache_misses": _judge_cache_misses,
                "cost_saved_usd_est": round(_judge_cost_saved_usd_est, 4),
            },
        }


