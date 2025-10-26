import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "rainy day commute, waterproof boots, budget 120", "limit": 10, "use_judge": False, "debug": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    tr = data.get("trace", {})
    if not tr:
        print("[FAIL] trace missing while debug=true")
        raise SystemExit(1)
    if tr.get("retrieval", {}).get("k") is None:
        print("[FAIL] retrieval.k missing")
        raise SystemExit(1)
    if tr.get("mmr", {}).get("final") != 10:
        print("[FAIL] mmr.final != limit")
        raise SystemExit(1)
    print("[OK] search_trace")


if __name__ == "__main__":
    main()


