import httpx


def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "gym set, no logos, breathable", "use_judge": False}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    slots = r.json().get("slots", {})
    for items in slots.values():
        for it in items:
            title = (it.get("title") or "").lower()
            if "logo" in title:
                print("[FAIL] found logo despite constraint")
                raise SystemExit(1)
    print("[OK] outfit_constraints")


if __name__ == "__main__":
    main()


