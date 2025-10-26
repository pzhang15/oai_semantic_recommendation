import argparse
import csv
import json
import os
import time
from datetime import datetime

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--url", default="http://localhost:8000/search")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--qps", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"search_results_{ts}.csv")
    out_json = os.path.join(args.outdir, f"search_summary_{ts}.json")

    rows = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    results = []
    latencies = []
    with httpx.Client(timeout=args.timeout) as client:
        for obj in rows:
            query = obj.get("query")
            limit = obj.get("limit", 12)
            use_judge = bool(obj.get("use_judge", False))
            payload = {"query": query, "limit": limit, "use_judge": use_judge, "debug": True}
            t0 = time.perf_counter()
            try:
                r = client.post(args.url, json=payload)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                if r.status_code != 200:
                    results.append({"query": query, "status": r.status_code, "error": r.text})
                else:
                    data = r.json()
                    items = data.get("items", [])
                    trace = data.get("trace", {})
                    timings = (trace.get("timings") or {}) if trace else {}
                    top = items[0] if items else {}
                    results.append({
                        "query": query,
                        "limit": limit,
                        "use_judge": use_judge,
                        "status": 200,
                        "latency_ms": round(dt, 2),
                        "parse_ms": timings.get("parse_ms"),
                        "judge_ms": timings.get("judge_ms"),
                        "mmr_ms": timings.get("mmr_ms"),
                        "top_id": top.get("id"),
                        "top_score": top.get("score"),
                        "top_price": top.get("price"),
                    })
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                results.append({"query": query, "status": 0, "error": str(e)})
            # throttle
            delay = 1.0 / max(0.1, args.qps)
            time.sleep(delay)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["query", "status"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    if results:
        lat = [r.get("latency_ms", 0) for r in results if r.get("latency_ms") is not None]
        lat_sorted = sorted(lat)
        def pct(vs, p):
            if not vs: return 0
            k = int((len(vs)-1) * p)
            return vs[k]
        summary = {
            "avg_latency_ms": round(sum(lat)/len(lat), 2) if lat else 0,
            "p50_ms": pct(lat_sorted, 0.5),
            "p95_ms": pct(lat_sorted, 0.95),
            "count": len(results),
        }
    else:
        summary = {"avg_latency_ms": 0, "p50_ms": 0, "p95_ms": 0, "count": 0}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] eval_search_batch -> {out_csv}, {out_json}")


if __name__ == "__main__":
    main()


