from __future__ import annotations

import time

import httpx


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    base = "http://127.0.0.1:8000"
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{base}/healthz")
        _assert(r.status_code == 200, "/healthz not OK")
        j = r.json()
        print("health:", j)

        # Basic search queries (avoid judge to keep it LLM-free)
        for q in [
            "navy linen shirt under $50",
            "dr martens boots not leather",
            "wedding guest dress blue",
        ]:
            t0 = time.perf_counter()
            resp = client.post(f"{base}/search", json={"query": q, "limit": 12, "use_judge": False, "debug": True})
            dt = (time.perf_counter() - t0) * 1000.0
            _assert(resp.status_code == 200, f"/search failed for '{q}'")
            js = resp.json()
            items = js.get("items", [])
            _assert(len(items) >= 3, f"Too few results for '{q}'")
            print(f"search '{q}' -> {len(items)} items in {dt:.1f}ms")

    print("Smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


