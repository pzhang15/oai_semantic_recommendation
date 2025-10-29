import json
import time

from fastapi.testclient import TestClient

from src.app import app


def test_search_component_timings() -> None:
    query = "men's navy jacket under $150"

    with TestClient(app) as client:
        # Warmup health check
        client.get("/healthz")

        t0 = time.perf_counter()
        resp = client.post(
            "/search",
            json={
                "query": query,
                "page": 1,
                "limit": 12,
                "use_judge": False,
                "debug": True,
            },
        )
        t1 = time.perf_counter()

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # End-to-end elapsed time
    e2e_ms = (t1 - t0) * 1000.0

    # Extract component timings emitted by the backend
    trace = data.get("trace", {})
    timings = trace.get("timings", {})

    required_keys = ["parse_ms", "retrieve_ms", "filter_ms", "judge_ms", "mmr_ms"]
    for k in required_keys:
        assert k in timings, f"missing timing: {k}"
        assert isinstance(timings[k], (int, float)), f"timing {k} not numeric"

    # Print a concise summary for manual inspection
    print("\n[QUERY]", query)
    print("[E2E_MS]", round(e2e_ms, 2))
    print("[COMPONENT_MS]", json.dumps({k: timings.get(k) for k in required_keys}, indent=2))

    # Sanity: each component should be non-negative, and sum should not exceed an exaggerated bound of end-to-end
    total_components = sum(float(timings.get(k, 0.0)) for k in required_keys)
    assert total_components >= 0.0
    # Allow overhead for HTTP, JSON, and app middleware. Bound at 2x e2e in case of clock skew.
    assert total_components <= e2e_ms * 2.0 + 50.0


