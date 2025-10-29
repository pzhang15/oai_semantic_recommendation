import argparse
import json
import statistics
import time
from typing import Any, Dict, List

import requests


DEFAULT_QUERIES: List[str] = [
    "office-appropriate summer outfit under $150, no logos, breathable, cotton/linen mix; size M top, 32x30 pants; navy/khaki",
    "men's waterproof hiking rain jacket, packable, under $120, avoid polyester, size medium, with pit zips",
    "women's black cocktail dress, knee-length, not bodycon, with sleeves, under $80, petite",
    "running shoes US 9.5 wide, neutral (not stability), budget $100, road, breathable mesh",
    "winter coat wool/cashmere blend, below-the-knee, dark green, under £200 GBP",
    "kids pajamas cotton only (no flame-retardant), 6T, space theme",
    "white dress shirt stain-resistant non-iron, slim fit, 15.5/34, under $60",
    "vegan leather boots (not leather), waterproof, lug sole, under $130",
    "merino base layer top 200-250 gsm, crew, itch-free, under $70",
    "linen trousers, relaxed fit, drawstring, off-white, under $65",
    "athleisure set: sports bra + high-waist leggings, seamless, squat-proof, under $85",
    "beach hat wide brim > 3in, UPF 50+, crushable, under $40",
    "wedding guest dress pastel floral, midi, sleeves, nursing-friendly, under $110",
    "denim jacket, oversized, light wash, no distressing, under $75",
    "travel backpack personal item size < 18x14x8, under 2 lb, under $90",
    "sneakers all-white, leather upper, minimal branding, under €120",
    "rain pants breathable (20k/20k), full side zips, under $140",
    "golf polo anti-odor, no big logo, tall sizes, under $55",
    "wool beanie non-itch (merino or cashmere), ribbed, under $35",
    "office heels 2-3 inch block heel, wide toe box, under $95",
]


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = int(0.95 * (len(sorted_vals) - 1))
    return float(sorted_vals[k])


def run(host: str, queries: List[str], limit: int, use_judge: bool) -> None:
    url = f"{host.rstrip('/')}/search"
    results: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        body = {
            "query": q,
            "limit": limit,
            "page": 1,
            "use_judge": use_judge,
            "debug": True,
        }
        t0 = time.perf_counter()
        resp = requests.post(url, json=body, timeout=120)
        t1 = time.perf_counter()
        e2e_ms = (t1 - t0) * 1000.0
        ok = resp.status_code == 200
        row: Dict[str, Any] = {
            "idx": i,
            "query": q,
            "status": resp.status_code,
            "e2e_ms": round(e2e_ms, 2),
            "count": 0,
            "parse_ms": None,
            "retrieve_ms": None,
            "filter_ms": None,
            "judge_ms": None,
            "mmr_ms": None,
        }
        if ok:
            try:
                data = resp.json()
                row["count"] = len(data.get("items", []))
                tr = data.get("trace", {}) or {}
                t = tr.get("timings", {}) or {}
                for k in ["parse_ms", "retrieve_ms", "filter_ms", "judge_ms", "mmr_ms"]:
                    v = t.get(k)
                    if isinstance(v, (int, float)):
                        row[k] = round(float(v), 2)
            except Exception:
                pass
        results.append(row)

    # Print table
    print("\nE2E runtime benchmark ({} queries)\n".format(len(results)))
    print("{:>3}  {:>8}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:<40}".format(
        "#", "e2e_ms", "count", "parse_ms", "retr_ms", "judge_ms", "mmr_ms", "query[:40]"
    ))
    for r in results:
        print("{:>3}  {:>8}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:<40}".format(
            r["idx"],
            f"{r['e2e_ms']:.2f}",
            r["count"],
            f"{(r['parse_ms'] or 0):.2f}",
            f"{(r['retrieve_ms'] or 0):.2f}",
            f"{(r['judge_ms'] or 0):.2f}",
            f"{(r['mmr_ms'] or 0):.2f}",
            r["query"][:40],
        ))

    e2e = [float(r["e2e_ms"]) for r in results]
    print("\nSummary:")
    print("- e2e avg:  {:.2f} ms".format(statistics.mean(e2e) if e2e else 0.0))
    print("- e2e p95:  {:.2f} ms".format(p95(e2e) if e2e else 0.0))
    # component summaries (when debug available)
    for comp in ["parse_ms", "retrieve_ms", "judge_ms", "mmr_ms"]:
        vals = [float(r[comp]) for r in results if isinstance(r.get(comp), (int, float))]
        if vals:
            print(f"- {comp} avg: {statistics.mean(vals):.2f} ms; p95: {p95(vals):.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run E2E runtime benchmark against /search")
    ap.add_argument("--host", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--use-judge", action="store_true")
    args = ap.parse_args()

    qs = DEFAULT_QUERIES[: args.n]
    if len(qs) < args.n:
        qs = (DEFAULT_QUERIES * ((args.n // len(DEFAULT_QUERIES)) + 1))[: args.n]
    run(args.host, qs, args.limit, args.use_judge)


if __name__ == "__main__":
    main()


