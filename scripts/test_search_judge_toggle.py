import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "smart-casual office look for women, neutral colors", "limit": 8, "use_judge": False}
    r1 = httpx.post(url, json=payload, timeout=60.0)
    if r1.status_code != 200:
        print(f"[ERROR] A status {r1.status_code}: {r1.text}")
        raise SystemExit(1)
    a = r1.json().get("items", [])

    payload2 = dict(payload)
    payload2["use_judge"] = True
    r2 = httpx.post(url, json=payload2, timeout=60.0)
    if r2.status_code != 200:
        print(f"[ERROR] B status {r2.status_code}: {r2.text}")
        raise SystemExit(1)
    b = r2.json().get("items", [])

    if len(a) < 8 or len(b) < 8:
        print("[FAIL] fewer than 8 items returned")
        raise SystemExit(1)

    order_diff = any((ai.get("id") != bi.get("id")) for ai, bi in zip(a, b))
    why_present = any((it.get("why") for it in b))
    if not order_diff and not why_present:
        print("[FAIL] judge appears to have no effect")
        raise SystemExit(1)
    print("[OK] search_judge_toggle")


if __name__ == "__main__":
    main()


