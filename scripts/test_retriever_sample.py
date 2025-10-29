import argparse
import json
import os
from pathlib import Path
import sys

# Ensure 'src' is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings  # type: ignore
from src.core.retriever import retrieve  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug retriever: embed → FAISS → join → items summary")
    ap.add_argument("--query", required=True, help="Natural-language query")
    ap.add_argument("--k", type=int, default=12, help="Top-K to request from FAISS")
    ap.add_argument("--require-price-image", choices=["true", "false"], default=None,
                    help="Override SEARCH_REQUIRE_PRICE_OR_IMAGE env for this run")
    ap.add_argument("--show", type=int, default=10, help="How many items to print in detail")
    args = ap.parse_args()

    # Optional override of quality gate
    if args.require_price_image is not None:
        os.environ["SEARCH_REQUIRE_PRICE_OR_IMAGE"] = args.require_price_image

    settings = get_settings()
    print(f"[SETTINGS] index={settings.index_path} parquet={settings.parquet_path}")
    print(f"[SETTINGS] SEARCH_REQUIRE_PRICE_OR_IMAGE={settings.search_require_price_or_image}")

    res = retrieve(args.query, k=args.k)
    items = res.get("items", [])
    print(f"\n[RESULT] items={len(items)} requested_k={args.k}")
    # Field completeness summary
    def has(v):
        return v not in (None, "", [])

    num_price = sum(1 for it in items if has(it.get("price")))
    num_image = sum(1 for it in items if has(it.get("image_url")))
    num_brand = sum(1 for it in items if has(it.get("brand")))
    print(f"[FIELDS] price={num_price}/{len(items)}  image={num_image}/{len(items)}  brand={num_brand}/{len(items)}")

    # Required rich fields expected on retrieved items
    required_fields = [
        "main_category",
        "title",
        "average_rating",
        "rating_number",
        "features",
        "description",
        "price",
        "images",
        "videos",
        "store",
        "categories",
        "details",
        "parent_asin",
        "bought_together",
    ]

    # Show first N items with full field dump and validate presence
    show_n = min(args.show, len(items))
    total_missing = {k: 0 for k in required_fields}
    for i in range(show_n):
        it = items[i]
        print(f"\n#{i+1}: id={it.get('id')}  score={it.get('score')}")
        # Dump all fields present on the item
        for k in sorted(it.keys()):
            try:
                v = it[k]
                js = json.dumps(v) if isinstance(v, (dict, list)) else v
            except Exception:
                js = str(it[k])
            print(f"  {k}: {js}")
        # Validate presence of required fields
        missing = [k for k in required_fields if k not in it]
        for k in missing:
            total_missing[k] += 1
        if missing:
            print(f"  [MISSING] {', '.join(missing)}")

    # Emit a compact JSON summary for programmatic check
    summary = {
        "counts": {
            "items": len(items),
            "with_price": num_price,
            "with_image": num_image,
            "with_brand": num_brand,
        },
        "missing_fields_firstN": total_missing,
        "trace": res.get("trace", {}),
    }
    print("\n[JSON]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


