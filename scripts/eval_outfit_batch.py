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
    ap.add_argument("--url", default="http://localhost:8000/outfit")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--qps", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"outfit_results_{ts}.csv")
    out_json = os.path.join(args.outdir, f"outfit_summary_{ts}.json")

    rows = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    results = []
    latencies = []
    with httpx.Client(timeout=args.timeout) as client:
        for obj in rows:
            payload = {"query": obj.get("query"), "use_judge": bool(obj.get("use_judge", False)), "debug": True}
            t0 = time.perf_counter()
            try:
                r = client.post(args.url, json=payload)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                if r.status_code != 200:
                    results.append({"query": payload["query"], "status": r.status_code, "error": r.text})
                else:
                    data = r.json()
                    slots = data.get("slots", {})
                    trace = data.get("trace", {})
                    timings = (trace.get("timings") or {}) if trace else {}
                    results.append({
                        "query": payload["query"],
                        "status": 200,
                        "latency_ms": round(dt, 2),
                        "compose_ms": timings.get("compose_ms"),
                        "total_price": data.get("total_price"),
                        "under_budget": data.get("under_budget"),
                        "slots_filled": sum(1 for k in ["top","bottom","shoes"] if slots.get(k)),
                    })
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                results.append({"query": payload["query"], "status": 0, "error": str(e)})
            delay = 1.0 / max(0.1, args.qps)
            time.sleep(delay)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["query", "status"])
        w.writeheader()
        for r in results:
            w.writerow(r)

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
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] eval_outfit_batch -> {out_csv}, {out_json}")


if __name__ == "__main__":
    main()


