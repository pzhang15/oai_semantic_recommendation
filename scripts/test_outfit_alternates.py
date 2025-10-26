import httpx


def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "smart casual office, women, neutral colors", "use_judge": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    slots = r.json().get("slots", {})
    has_alt = any(len(items) >= 2 for items in slots.values())
    if not has_alt:
        print("[FAIL] no alternates found")
        raise SystemExit(1)
    titles = []
    for items in slots.values():
        if len(items) >= 2:
            titles = [i.get("title") for i in items[:2]]
            break
    if titles and titles[0] == titles[1]:
        print("[FAIL] alternates are duplicates")
        raise SystemExit(1)
    print("[OK] outfit_alternates")


if __name__ == "__main__":
    main()


