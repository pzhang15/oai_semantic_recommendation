from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

import requests


BASE = "http://localhost:8000"
OUT = Path("docs/design/metrics.json")

QUERIES = [
    "navy linen shirt under $50",
    "dr martens boots not leather",
    "wedding guest dress blue",
]


def main() -> None:
    lat_ms: list[float] = []
    for q in QUERIES:
        t0 = time.perf_counter()
        # API expects key "query"
        r = requests.post(f"{BASE}/search", json={"query": q}, timeout=30)
        r.raise_for_status()
        dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)

    health = requests.get(f"{BASE}/healthz", timeout=5).json()

    p50 = stats.median(lat_ms) if lat_ms else 0.0
    p95 = sorted(lat_ms)[min(len(lat_ms) - 1, max(0, int(round(len(lat_ms) * 0.95)) - 1))] if lat_ms else 0.0

    out = {"health": health, "lat_ms": lat_ms, "p50": round(p50, 2), "p95": round(p95, 2)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


