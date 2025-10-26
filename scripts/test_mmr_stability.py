import httpx


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "black cotton t-shirt", "limit": 12, "use_judge": False}
    r1 = httpx.post(url, json=payload, timeout=60.0)
    r2 = httpx.post(url, json=payload, timeout=60.0)
    if r1.status_code != 200 or r2.status_code != 200:
        print(f"[ERROR] status A {r1.status_code} B {r2.status_code}")
        raise SystemExit(1)
    a = [it.get("id") for it in r1.json().get("items", [])]
    b = [it.get("id") for it in r2.json().get("items", [])]
    if a != b:
        print("[FAIL] non-deterministic ordering")
        raise SystemExit(1)
    print("[OK] mmr_stability")


if __name__ == "__main__":
    main()


