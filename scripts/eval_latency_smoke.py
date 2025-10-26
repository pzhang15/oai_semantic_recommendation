import argparse
import time
import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/search")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    queries = [
        "beach trip under $120",
        "smart casual office women",
        "rainy day commute waterproof boots",
        "black cotton t-shirt",
        "festival outfit under $150",
    ]
    rows = []
    with httpx.Client(timeout=args.timeout) as client:
        for q in queries:
            t0 = time.perf_counter()
            try:
                r = client.post(args.url, json={"query": q, "limit": 8, "use_judge": False})
                dt = (time.perf_counter() - t0) * 1000.0
                n = len(r.json().get("items", [])) if r.status_code == 200 else 0
                rows.append((q, str(r.status_code), dt, n))
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                rows.append((q, "TIMEOUT", dt, 0))
    for q, st, dt, n in rows:
        print(f"{st:>7}  {dt:7.1f} ms  n={n:2d}  {q}")
    # Consider success if at least one request succeeded
    ok = any(st.isdigit() and int(st) == 200 for (_, st, _, _) in rows)
    print("[OK] latency_smoke" if ok else "[WARN] latency_smoke had failures")


if __name__ == "__main__":
    main()


