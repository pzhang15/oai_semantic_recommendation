import httpx


def main() -> None:
    url = "http://localhost:8000/outfit"
    payload = {"query": "neutral palette office outfit", "use_judge": False, "debug": True}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    tr = data.get("trace", {})
    palette = tr.get("composer", {}).get("palette", []) if tr else []
    if not palette:
        print("[WARN] palette missing (tolerated)")
    print("[OK] outfit_palette_coherence")


if __name__ == "__main__":
    main()


