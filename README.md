# Semantic Fashion Recs — FAISS + OpenAI (parser/judge/outfit)

Semantic search and outfit composition service using FAISS (IndexFlatIP) and OpenAI for parsing, judging, and composing. Telemetry captures token usage, timings, and estimated cost.

## Highlights
- Semantic queries (e.g., "beach trip under $120", "smart casual office outfit")
- FAISS exact search (IndexFlatIP) on ~826k items (1536-D unit vectors)
- LLM parser with Structured Outputs; judge reranker (batch + caching)
- MMR diversity + outfit composer (slot-based)
- Telemetry with token/latency/cost; `/debug/stats` and eval scripts

## Architecture
![Architecture](docs/architecture.png)

See sequences: [search](docs/search-sequence.png) | [outfit](docs/outfit-sequence.png)

## Prerequisites
- Python 3.11
- Windows note: FAISS requires NumPy < 2. Install FAISS from conda-forge: `conda install -c conda-forge faiss-cpu`
- OpenAI API key in environment or `.env`

## Quickstart
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt

# Create .env (set OPENAI_API_KEY and optional MODEL_*)
# Build embeddings and FAISS (see scripts/index.py options)
python scripts/index.py --embed
python scripts/index.py --build-faiss

# Run API
uvicorn src.app:app --reload
```

Health:
```bash
curl http://127.0.0.1:8000/healthz
```

## Environment variables
- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional)
- `MODEL_EMBED` default `text-embedding-3-small`
- `MODEL_PARSER` default `gpt-4o-mini`
- `MODEL_JUDGE` default = `MODEL_PARSER`
- `INDEX_PATH`, `PARQUET_PATH`, `K_RETRIEVE`, `TOP_K_DEFAULT`, `MMR_LAMBDA`, `USE_JUDGE_DEFAULT`
- Outfit: `OUTFIT_*` (slots, pool size, timeouts, tolerance)

Note: temperature=0 is used when supported; we auto-fallback if the model rejects custom temperature.

## Endpoints
- `GET /healthz`
- `POST /debug/retrieve` body: `{ "query": "...", "k": 10 }`
- `POST /debug/parse` body: `{ "query": "..." }`
- `POST /search` body: `{ "query": "...", "limit": 12, "use_judge": true }`
- `POST /outfit` body: `{ "query": "...", "use_judge": true }`
- `GET /debug/stats`

Examples:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"linen shirt under $60","limit":12,"use_judge":false}'

curl -X POST http://127.0.0.1:8000/outfit \
  -H "Content-Type: application/json" \
  -d '{"query":"summer beach outfit under $150, light colors, linen","use_judge":true}'
```

## Design decisions & trade-offs
- OpenAI-heavy vs local: faster iteration; Structured Outputs for determinism; schema validation; fallbacks
- FAISS vs in-memory NumPy: 826k x 1536D ~4.7GB float32; FAISS gives C++ speed and easy ANN path
- IndexFlatIP now vs ANN later: quality first; API is switchable
- MMR & variant control: vector cosine + title Jaccard; hard cap on variants
- Budget: filters & judge penalty; outfit finalizer recomputes totals and ensures numeric total_price
- Resilience: retries and temperature fallback; judge failure -> retrieval-only

## Evaluation & diagnostics
Batch eval:
```bash
python scripts/eval_search_batch.py --in samples/queries_search.jsonl --out reports/
python scripts/eval_outfit_batch.py  --in samples/queries_outfit.jsonl --out reports/
python scripts/eval_cost_snapshot.py --out reports/
```
Smoke:
```bash
python scripts/eval_latency_smoke.py
```
Stats:
```bash
curl http://127.0.0.1:8000/debug/stats
```

## Troubleshooting
- Windows FAISS + NumPy: ensure `numpy>=1.26,<2`; install FAISS via conda-forge
- OpenAI errors: temperature not supported -> auto-retry without temperature
- Parser 422: reduce prompt complexity; we retry with validation hints
- Index mismatch: ensure `ids.json` length equals `index.ntotal`

## Repo layout
- `src/` (app/core/models/routers/telemetry)
- `scripts/` (index, smoke tests, eval, render_diagrams)
- `docs/` (mermaid + png/pdf)
- `data/` (ignored in VCS)
- `samples/` (queries)

## Diagrams
See `docs/README-diagrams.md`. Render with:
```bash
make diagrams
```

## License / Credits
MIT. Dataset attribution per assignment; built for evaluation via CodexCLI.

