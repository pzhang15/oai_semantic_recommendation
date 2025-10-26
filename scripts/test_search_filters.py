import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "women's office outfit, avoid logos, under $150", "limit": 10, "use_judge": False, "debug": True}
    resp = httpx.post(url, json=payload, timeout=60.0)
    if resp.status_code != 200:
        print(f"[ERROR] status {resp.status_code}: {resp.text}")
        raise SystemExit(1)
    data = resp.json()
    items = data.get("items", [])
    for it in items:
        title = (it.get("title") or "").lower()
        if "logo" in title:
            print("[FAIL] item contains 'logo' in title")
            raise SystemExit(1)
        if "men" in title or "men's" in title or "mens" in title:
            print("[FAIL] item appears to be men's")
            raise SystemExit(1)
    tr = data.get("trace", {})
    if tr:
        kept = tr.get("filters", {}).get("kept", 0)
        dropped = tr.get("filters", {}).get("dropped", 0)
        hits = tr.get("retrieval", {}).get("hits", 0)
        if kept + dropped < hits * 0.8:  # tolerate judge/MMR differences
            print("[WARN] kept+dropped << hits (tolerated)")
    print("[OK] search_filters")


if __name__ == "__main__":
    main()


