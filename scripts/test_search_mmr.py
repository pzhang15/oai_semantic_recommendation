import re
import httpx


def stem_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(t.split()[:5])


def main() -> None:
    url = "http://localhost:8000/search"
    payload = {"query": "black cotton t-shirt", "limit": 12, "use_judge": False}
    r = httpx.post(url, json=payload, timeout=60.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    items = r.json().get("items", [])
    stems = {}
    for it in items:
        st = stem_title(it.get("title"))
        stems[st] = stems.get(st, 0) + 1
    if any(c > 2 for c in stems.values()):
        print("[FAIL] too many similar titles (MMR not effective)")
        raise SystemExit(1)
    print("[OK] search_mmr")


if __name__ == "__main__":
    main()


