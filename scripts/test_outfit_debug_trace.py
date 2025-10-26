import httpx


def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "beach trip outfit under $120", "use_judge": False, "debug": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    tr = r.json().get("trace", {})
    if not tr:
        print("[FAIL] trace missing")
        raise SystemExit(1)
    comp = tr.get("composer", {})
    if not comp.get("model") or comp.get("pool") is None:
        print("[FAIL] composer trace fields missing")
        raise SystemExit(1)
    req = comp.get("required_slots", [])
    if not req or not all(s in req for s in ["top", "bottom", "shoes"]):
        print("[FAIL] required_slots invalid")
        raise SystemExit(1)
    print("[OK] outfit_debug_trace")


if __name__ == "__main__":
    main()


