import httpx


def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "summer beach outfit under $150, light colors, linen", "use_judge": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    slots = data.get("slots", {})
    for req in ["top", "bottom", "shoes"]:
        if not slots.get(req):
            print(f"[FAIL] missing required slot: {req}")
            raise SystemExit(1)
    if not data.get("summary"):
        print("[FAIL] summary missing")
        raise SystemExit(1)
    if "under_budget" not in data:
        print("[FAIL] under_budget missing")
        raise SystemExit(1)
    print("[OK] outfit_basic")


if __name__ == "__main__":
    main()


