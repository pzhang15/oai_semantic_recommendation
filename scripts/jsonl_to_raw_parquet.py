import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def iter_json_objects(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line in ("[", "]", "],", ","):
                continue
            if line.endswith(","):
                line = line[:-1].rstrip()
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def convert_jsonl_to_parquet(input_path: Path, output_path: Path, batch_size: int, compression: str, max_rows: int | None) -> int:
    schema = pa.schema([pa.field("json", pa.large_string())])
    writer = pq.ParquetWriter(output_path, schema=schema, compression=compression)
    batch: list[str] = []
    count = 0
    try:
        for obj in iter_json_objects(input_path):
            batch.append(json.dumps(obj, ensure_ascii=False))
            if len(batch) >= batch_size:
                tbl = pa.Table.from_arrays([pa.array(batch, type=pa.large_string())], schema=schema)
                writer.write_table(tbl)
                count += len(batch)
                print(f"wrote: {count}")
                batch = []
            if max_rows is not None and (count + len(batch)) >= max_rows:
                break
        if batch:
            tbl = pa.Table.from_arrays([pa.array(batch, type=pa.large_string())], schema=schema)
            writer.write_table(tbl)
            count += len(batch)
    finally:
        writer.close()
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSONL to raw Parquet (single 'json' column)")
    ap.add_argument("--input", type=str, default="data/meta_Amazon_Fashion.jsonl", help="Path to input JSONL")
    ap.add_argument("--output", type=str, default="data/raw_products.parquet", help="Path to output Parquet")
    ap.add_argument("--batch-size", type=int, default=100_000, help="Rows per parquet write batch")
    ap.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional limit to rows processed")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    total = convert_jsonl_to_parquet(input_path, output_path, args.batch_size, args.compression, args.max_rows)
    print(f"Total rows written: {total}")


if __name__ == "__main__":
    main()


