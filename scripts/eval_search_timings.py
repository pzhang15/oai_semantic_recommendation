import argparse
import json
import time
from typing import Dict, List

import httpx


COMP_KEYS = ["parse_ms", "retrieve_ms", "filter_ms", "judge_ms", "mmr_ms"]


def main() -> None:
  ap = argparse.ArgumentParser(description="Measure per-component timings for a search query")
  ap.add_argument("--url", default="http://localhost:8000/search", help="Search endpoint URL")
  ap.add_argument("--query", default="men's navy jacket under $150", help="Query text")
  ap.add_argument("--page", type=int, default=1)
  ap.add_argument("--limit", type=int, default=12)
  ap.add_argument("--no-judge", action="store_true", help="Disable judge stage")
  ap.add_argument("--repeats", type=int, default=1, help="Number of repeated runs")
  ap.add_argument("--timeout", type=float, default=90.0)
  ap.add_argument("--delay", type=float, default=0.25, help="Delay between repeats in seconds")
  args = ap.parse_args()

  rows: List[Dict[str, float]] = []
  with httpx.Client(timeout=args.timeout) as client:
    for i in range(args.repeats):
      t0 = time.perf_counter()
      r = client.post(
        args.url,
        json={
          "query": args.query,
          "page": args.page,
          "limit": args.limit,
          "use_judge": (not args.no_judge),
          "debug": True,
        },
      )
      e2e_ms = (time.perf_counter() - t0) * 1000.0
      if r.status_code != 200:
        print(f"ERROR {r.status_code}: {r.text}")
        return
      data = r.json()
      timings = (data.get("trace", {}) or {}).get("timings", {})
      row = {k: float(timings.get(k, 0.0)) for k in COMP_KEYS}
      row["e2e_ms"] = e2e_ms
      rows.append(row)
      if i < args.repeats - 1 and args.delay > 0:
        time.sleep(args.delay)

  # Print per-run
  print(f"\nQUERY: {args.query}")
  for i, row in enumerate(rows, 1):
    comps = " ".join([f"{k}={row.get(k, 0.0):.2f}ms" for k in COMP_KEYS])
    print(f"run#{i}: e2e={row['e2e_ms']:.2f}ms  {comps}")

  # Averages
  if len(rows) > 1:
    avg = {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in [*COMP_KEYS, "e2e_ms"]}
    print("\navg:", json.dumps(avg, indent=2))


if __name__ == "__main__":
  main()


