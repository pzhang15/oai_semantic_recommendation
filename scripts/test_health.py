import sys
import httpx


def main() -> None:
    url = "http://localhost:8000/healthz"
    try:
        resp = httpx.get(url, timeout=10.0)
    except Exception as e:
        print(f"[ERROR] Failed to GET {url}: {e}")
        raise SystemExit(1)

    if resp.status_code != 200:
        print(f"[ERROR] Unexpected status: {resp.status_code}")
        raise SystemExit(1)

    data = resp.json()
    for key in ["backend", "dim", "size", "index_type"]:
        if key not in data:
            print(f"[ERROR] Missing key in healthz: {key}")
            raise SystemExit(1)
    if data["backend"] != "faiss":
        print(f"[ERROR] backend != faiss: {data['backend']}")
        raise SystemExit(1)

    print(f"[OK] healthz: backend={data['backend']} dim={data['dim']} size={data['size']} index_type={data['index_type']}")


if __name__ == "__main__":
    main()


