import httpx
import json

def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "outfit under $90", "use_judge": False}
    r = httpx.post(url, json=payload, timeout=60.0)
    print(r.status_code, json.dumps(r.json(), indent=2))
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    total = data.get("total_price")
    under = data.get("under_budget")
    if total is None or under is None:
        print("[FAIL] total/under_budget missing")
        raise SystemExit(1)
    print("[OK] outfit_budget")


if __name__ == "__main__":
    main()


