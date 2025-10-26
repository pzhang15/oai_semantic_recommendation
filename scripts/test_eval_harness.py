import httpx


def main() -> None:
    s_url = "http://localhost:8000/search"
    o_url = "http://localhost:8000/outfit"
    d_url = "http://localhost:8000/debug/stats"

    # Baseline stats
    r0 = httpx.get(d_url, timeout=10.0)
    base_calls = (r0.json().get("components", {}).get("parser", {}).get("calls", 0)) if r0.status_code == 200 else 0

    # Two search
    for q in ["linen shirt under $60", "black t-shirt"]:
        rs = httpx.post(s_url, json={"query": q, "limit": 8, "use_judge": False}, timeout=30.0)
        if rs.status_code != 200:
            print("[FAIL] search status", rs.status_code)
            raise SystemExit(1)

    # One outfit
    ro = httpx.post(o_url, json={"query": "beach outfit under $120"}, timeout=30.0)
    if ro.status_code != 200:
        print("[FAIL] outfit status", ro.status_code)
        raise SystemExit(1)

    r1 = httpx.get(d_url, timeout=10.0)
    now_calls = (r1.json().get("components", {}).get("parser", {}).get("calls", 0)) if r1.status_code == 200 else 0
    if now_calls <= base_calls:
        print("[FAIL] counters did not increase")
        raise SystemExit(1)

    print("[OK] eval harness")


if __name__ == "__main__":
    main()


