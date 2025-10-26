import httpx


def main() -> None:
    url = "http://localhost:8000/debug/retrieve"
    payload = {"query": "smart casual office outfit", "k": 7}
    try:
        resp = httpx.post(url, json=payload, timeout=20.0)
    except Exception as e:
        print(f"[ERROR] Failed to POST {url}: {e}")
        raise SystemExit(1)

    if resp.status_code != 200:
        print(f"[ERROR] Unexpected status: {resp.status_code} body={resp.text}")
        raise SystemExit(1)

    data = resp.json()
    items = data.get("items", [])
    if len(items) != 7:
        print(f"[ERROR] Expected 7 items; got {len(items)}")
        raise SystemExit(1)
    for it in items:
        if not it.get("id") or not it.get("title"):
            print("[ERROR] Missing id/title in item")
            raise SystemExit(1)
        sc = it.get("score")
        if sc is None or sc < 0:
            print("[ERROR] Invalid score in item")
            raise SystemExit(1)

    print("[OK] debug/retrieve returned 7 items with ids, titles, non-negative scores")


if __name__ == "__main__":
    main()


