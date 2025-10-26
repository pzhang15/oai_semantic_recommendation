import httpx


def main() -> None:
    url = "http://localhost:8000/debug/parse"
    payload = {"query": "neutral palette, summer wedding guest dress under $200"}
    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
    except Exception as e:
        print(f"[ERROR] Failed to POST {url}: {e}")
        raise SystemExit(1)

    if resp.status_code != 200:
        print(f"[ERROR] Unexpected status: {resp.status_code} body={resp.text}")
        raise SystemExit(1)

    data = resp.json()
    facets = data.get("facets", {})
    if not facets:
        print("[ERROR] Missing facets in response")
        raise SystemExit(1)
    intent = facets.get("intent")
    if intent not in {"search", "outfit"}:
        print(f"[ERROR] Unexpected intent: {intent}")
        raise SystemExit(1)
    budget = facets.get("budget_usd", {})
    mx = budget.get("max")
    if mx is None or mx > 200 + 1e-6:
        print("[ERROR] Budget max > 200")
        raise SystemExit(1)
    occ = facets.get("occasion", "")
    if "wedding" not in str(occ):
        print("[WARN] occasion may not include wedding (not failing)")
    print("[OK] parse_route")


if __name__ == "__main__":
    main()


