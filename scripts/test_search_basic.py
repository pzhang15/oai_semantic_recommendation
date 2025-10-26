import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "linen shirt under $60", "limit": 8, "use_judge": False}
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
    except Exception as e:
        print(f"[ERROR] POST {url} failed: {e}")
        raise SystemExit(1)
    if resp.status_code != 200:
        print(f"[ERROR] status {resp.status_code}: {resp.text}")
        raise SystemExit(1)
    data = resp.json()
    items = data.get("items", [])
    if len(items) != 8:
        print(f"[FAIL] expected 8 items, got {len(items)}")
        raise SystemExit(1)
    if any((it.get("score") or 0) < 0 for it in items):
        print("[FAIL] negative scores present")
        raise SystemExit(1)
    print("[OK] search_basic")


if __name__ == "__main__":
    main()


