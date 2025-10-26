import argparse
import json
import os
from datetime import datetime

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--url", default="http://localhost:8000/debug/stats")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(args.outdir, f"cost_snapshot_{ts}.json")

    r = httpx.get(args.url, timeout=10.0)
    if r.status_code != 200:
        print(f"[ERROR] status {r.status_code}: {r.text}")
        raise SystemExit(1)
    data = r.json()
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # One-line summary
    models = data.get("models", {})
    total_cost = sum((m.get("cost_usd_est", 0.0) or 0.0) for m in models.values())
    print(f"[OK] cost_snapshot -> {outfile}  est_cost=${total_cost:.2f}")


if __name__ == "__main__":
    main()


