## Semantic Fashion Recs — FAISS + OpenAI (parser/judge)

Semantic product search with FAISS for dense retrieval and optional OpenAI features (parser, judge). Ships with a one‑click Docker setup and sensible fallbacks when no API key is provided.

### Highlights
- Exact dense retrieval (FlatIP on unit vectors) + lexical recall
- Deterministic facet parsing first, with LLM fallback only on low‑confidence
- Micro query expansion (a few controlled rewrites; no heavy paraphrasing)
- Hybrid fusion and post‑processing: RRF → priors/filters → MMR diversity
- Gated second stage: SKIP/CHEAP/FULL with rubric‑driven judge and cache
- Observability built‑in: `/healthz`, sub‑timers, and debug routes
- Docker one‑click: api (FastAPI/Uvicorn) + web (Vite), LLM‑off demo mode supported

Core components:
- Constraint‑first understanding (deterministic facets: price/category/material/brand)
- Hybrid retrieval (dense + TF‑style lexical) with controlled expansions
- Rank fusion (RRF), priors/filters enforcement, and MMR diversity control
- Judge gate to re‑rank only when it measurably helps (cached by signature)
- Telemetry and health reporting for credible tuning and easy troubleshooting

Star feature:
- **Fast, explainable search that respects constraints first and only spends on re‑ranking when it moves the needle.** The gate keeps p50 low and costs predictable, while “why‑chips” and sub‑timers make the system trustworthy and easy to iterate.

### Getting Started (Docker, ZIP users)
This is the fastest way to run the demo from a zip with no prior knowledge.

Prerequisites:
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)

Steps:
1) Unzip the package to a folder (e.g., `semantic-rec/`).
2) Create `.env` from the example:
   - Windows PowerShell:
     ```powershell
     Copy-Item env.example .env -Force
     ```
   - macOS/Linux:
     ```bash
     cp env.example .env
     ```

3) Start the stack:
   ```bash
   docker compose up --build
   ```
   - Web: `http://localhost:5173`
   - API: `http://localhost:8000` (health: `/healthz`)

What happens on first run:
- API container runs a preflight check and a bootstrap step:
  - Verifies FAISS/TF‑IDF/parquet artifacts; logs warnings if missing
  - Builds TF‑IDF automatically if `data/products.parquet` exists
  - Does not auto‑build FAISS (embeddings require an API key), but the app will run in lexical‑only mode
- Web container installs Node modules inside the container (Linux‑native), avoiding Windows/macOS conflicts

### Architecture Diagram

- **PDF:** [docs/architecture/diagram.pdf](docs/architecture/diagram.pdf)
- **PNG:** [docs/architecture/diagram.png](docs/architecture/diagram.png)

Source: [`docs/architecture/diagram.mmd`](docs/architecture/diagram.mmd)

Re-render locally:
```bash
# Linux/macOS
./scripts/arch.sh

# Windows PowerShell
./scripts/arch.ps1

# or
make arch
```

## Key Design Decisions & Trade-offs

See **[docs/design/decisions.md](docs/design/decisions.md)** (with summary matrix).

## Additional Exploration

See **[docs/exploration/additional_exploration.md](docs/exploration/additional_exploration.md)** for notebooks, experimental scripts, improvements made, and next steps.

 ### Environment variables
 - `OPENAI_API_KEY` (optional): enables embeddings, parser, judge. If missing, the app runs in lexical‑only mode.
 - `MODEL_EMBED` (default: `text-embedding-3-small`)
 - `MODEL_PARSER` (default: `gpt-4o-mini`), `MODEL_JUDGE` (default: same as parser)
 - Index and retrieval: `INDEX_PATH`, `PARQUET_PATH`, `K_RETRIEVE`, `TOP_K_DEFAULT`, `MMR_LAMBDA`, `USE_JUDGE_DEFAULT`

 ### Endpoints
 - `GET /healthz`
 - `POST /search` body: `{ "query": "...", "limit": 12, "use_judge": true }`

Examples:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"linen shirt under $60","limit":12,"use_judge":false}'
```

### Optional: rebuild artifacts
You usually do not need these. Use them in clean environments or when experimenting.

- Rebuild lexical TF‑IDF (no API key needed):
  ```bash
  docker compose exec api bash -lc "python scripts/build_lexical.py"
  ```
- Build embeddings and FAISS (requires `OPENAI_API_KEY`):
  ```bash
  docker compose exec api bash -lc "python scripts/index.py --embed --build-faiss"
  ```
- Create an ANN variant from existing embeddings:
  ```bash
  docker compose exec api bash -lc "python scripts/build_index_ann.py --mode hnsw_pca \
    --emb data/embeddings.npy --ids data/ids.json \
    --out data/index_hnsw_pca.faiss --meta data/index_hnsw_pca.meta.json --dim_out 256"
  ```
- To use a non‑default index, set `INDEX_PATH` in `.env` and restart.

### Troubleshooting
- Web won’t start or loops on npm: remove any host `frontend/node_modules` and restart. The container now owns `node_modules`.
  - Windows PowerShell: `rd /s /q frontend\node_modules`
  - macOS/Linux: `rm -rf frontend/node_modules`
- OpenAI base URL errors: for standard OpenAI, keep `OPENAI_BASE_URL` unset. If using Azure/compatible endpoints, include `https://`.
- View logs: `docker compose logs -f api` and `docker compose logs -f web`
- Stop and clean: `docker compose down -v`

### Quickstart (local Python, optional)
For contributors who prefer running the API locally without Docker.

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt

# Create .env (set OPENAI_API_KEY if you want dense/LLM features)
cp env.example .env

# Run API
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/healthz
```

### Repo layout
- `src/` app, core, models, routers, telemetry
- `scripts/` indexing, eval, smoke, diagram rendering
- `docs/` architecture and sequence diagrams
- `data/` indexes and artifacts (generated/ignored)
- `frontend/` Vite + React UI

### License
MIT


