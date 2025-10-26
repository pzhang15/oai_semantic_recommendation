import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "black t-shirt", "limit": 12, "use_judge": False, "debug": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    items = data.get("items", [])
    tr = data.get("trace", {})
    mmr = tr.get("mmr", {}) if tr else {}
    if not items:
        print("[FAIL] no items returned")
        raise SystemExit(1)
    if mmr.get("final") != len(items):
        print("[FAIL] mmr.final mismatch")
        raise SystemExit(1)
    if "lambda" not in mmr or "mode" not in mmr:
        print("[FAIL] mmr trace missing fields")
        raise SystemExit(1)
    # Loose diversity check
    stems = {}
    for it in items:
        stem = (it.get("title") or "").lower().split(" ")[:5]
        stem = " ".join(stem)
        stems[stem] = stems.get(stem, 0) + 1
    if any(c > 3 for c in stems.values()):
        print("[WARN] possible low diversity (tolerated)")
    print("[OK] mmr_integration_route")


if __name__ == "__main__":
    main()


