# Additional Exploration & Improvements

Last updated: 2025-10-30 • Scope: data exploration, prompt/LLM experiments, retrieval/ranking trials, and system hardening.

## 1) Notebooks (EDA and data shaping)

- `notebooks/raw_to_parquet.ipynb`: Ingestion and schema normalization from raw JSONL/CSV to a consistent Parquet schema with stable fields (id, title, brand, categories, features, price, image_url, etc.). This notebook informed the production `scripts/index.py` normalization routines.
- `notebooks/raw_parquet_eda.ipynb`: Column profiling, null/value distributions, and basic quality checks (e.g., price ranges, title length). Findings drove the choice of robust parsing for price and brand inference.
- `notebooks/data_quality_eda.ipynb`: Sanity checks on duplicates, basic joinability across artifacts, and sampling for manual audit to validate the normalization and filtering rules.

Key outcomes: a stable Parquet schema, validated assumptions about price/brand sparsity, and inputs to deterministic parsers and lexical artifact selection.

## 2) Scripts used during experimentation

- Indexing & artifacts
  - `scripts/index.py`: End-to-end pipeline switches (`--embed`, `--build-faiss`, lexical build) exercised in isolation to compare candidate quality and timings.
  - `scripts/build_index_ann.py`: Measured ANN variants in a branch to quantify recall/latency trade-offs; kept exact retrieval in mainline.
  - `scripts/build_lexical.py`: Standalone lexical build to validate TF-style artifacts and field selections.

- Evaluation & smoke
  - `scripts/eval_search_batch.py`, `scripts/eval_outfit_batch.py`, `scripts/eval_cost_snapshot.py`, `scripts/eval_latency_smoke.py`: Batch and smoke harnesses used to confirm stability of hybrid, gate mix, and cost/latency caps.
  - `scripts/smoke_tests.py`: Quick regression checks across `/healthz` and core paths.

- Observability & diagrams
  - `scripts/collect_metrics.py` (added): Pulls `/healthz` and times three representative queries; produces `docs/design/metrics.json` for easy paste into docs.
  - `scripts/render_diagrams.py` + `scripts/arch.(sh|ps1)`: Re-producible architecture diagrams (PDF/PNG) from Mermaid sources.

## 3) LLM parsing & prompt exploration

- Deterministic-first parsing: Verified boundary/negation windows and alias maps for colors/materials/brands/prices; tuned thresholds so ambiguous cases fall back rather than produce incorrect hard constraints.
- LLM fallback: Iterated on a compact rubric for the fallback parser to reduce variability and keep token cost bounded; confirmed cache keys include version/rubric where applicable.

## 4) Retrieval, fusion, and ranking trials

- Hybrid retrieval: Compared dense-only vs. lexical-only vs. fused (rank-based) lists on a set of curated queries; adopted fusion for robustness across tail and synonym cases.
- Micro query expansion: Tested a small set of controlled expansions (aliases/taxonomy) under a strict time/variant budget; validated incremental recall without semantic drift.
- Post-processing: Tuned MMR strength conservatively to avoid near-duplicates while keeping the top cluster competitive; validated budget/category filters first to honor constraints.
- Gated re-rank: Calibrated a multi-signal gate (ambiguity, margins) for SKIP/CHEAP/FULL modes; verified caching eliminates repeat costs on steady inputs.

## 5) System hardening & DX improvements

- Docker one-click: Simplified `docker-compose.yml` (no fragile health-gating on web), ensured the Vite dev server binds to `0.0.0.0`, and mounted `node_modules` inside the container to avoid host OS binary issues.
- API robustness: Guarded base URL handling for the LLM client to ignore invalid values; improved degraded-mode behavior while preserving correctness.
- Documentation: Rewrote `README.md` for first-time users, added a high-level `docs/architecture/diagram.mmd` (v2) with render helpers, and created `docs/design/decisions.md` focused on overall design trade-offs and metrics to watch.

## 6) What we would do with more time

- Learned gate controller: Replace heuristic thresholds with a lightweight classifier trained on offline wins/losses to stabilize p95 and cost while improving win-rate.
- Fielded lexical upgrade: Move to BM25F or learned field weights for improved tail precision while keeping latency flat.
- Neural sparse signal: Introduce sparse neural retrieval to complement dense/lexical with minimal latency cost.
- Exact dimensionality reduction: Evaluate PCA-to-256D exact flow to reduce memory/bandwidth while maintaining cosine quality.
- CHEAP re-ranker: Add a tiny cross-encoder for CHEAP mode; keep the LLM judge for FULL path explanations and tie-breaks.
- Better expansions: Bandit policy for expansion selection under strict budgets; retire low-yield patterns automatically.
- A/B harness: Formalize evaluation with fixed query sets and tracked metrics (Recall@K proxy, violation rate, p50/p95, and gate mix) in CI.

## 7) Pointers

- Notebooks: `notebooks/raw_to_parquet.ipynb`, `notebooks/raw_parquet_eda.ipynb`, `notebooks/data_quality_eda.ipynb`
- Key scripts: `scripts/index.py`, `scripts/build_lexical.py`, `scripts/build_index_ann.py`, `scripts/eval_*`, `scripts/smoke_tests.py`, `scripts/collect_metrics.py`
- Artifacts: see `/healthz` for index type/size/dim and lexical readiness; metrics snapshot in `docs/design/metrics.json`


