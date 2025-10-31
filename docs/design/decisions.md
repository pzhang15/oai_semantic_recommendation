# Key Design Decisions & Trade-offs

Last updated: 2025-10-30 • System version: v2

This document explains the overall design choices and their trade-offs. It focuses on product behavior (what/why), not on specific libraries. Facts are derived from the codebase and `/healthz`.

For each decision:
- Why (user-centric)
- Trade-offs (engineering)
- Mitigations (how we offset costs)
- Metrics to watch (what proves success/regression)

---

## 1) Constraint‑first understanding

Decision: Parse constraints (price, category, material, brand) deterministically first; fall back only when confidence is low or the query is ambiguous.

- Why
  - Users expect filters (budget/category) to be honored before style; trust and predictability trump cleverness.
  - Deterministic parsing gives consistent behavior and avoids surprise re-interpretations of the query.
- Trade-offs
  - May miss nuanced intents without fallback.
  - Slightly lower recall when only deterministic logic is applied.
- Mitigations
  - Targeted fallback only when confidence is low; keep a strict budget and cache results.
  - Maintain alias maps and negation handling to reduce false positives.
- Metrics to watch
  - Monitor the parse confidence distribution, the fallback rate, the constraint violation rate within the top-N results, and the coverage of user-visible “why” chips.
  - This decision increases bias toward explicit constraints (which is desirable for trust) and reduces variance in interpretation, but it can lower recall for ambiguous phrasing if the fallback threshold is too strict.
  - It has very low CPU cost and improves end-to-end p50 latency by avoiding unnecessary heavy parsing, although an overly aggressive fallback threshold can hurt the recall–latency trade-off by triggering more fallbacks.

---

## 2) Hybrid retrieval

Decision: Combine complementary signals (dense and lexical) instead of relying on a single source.

- Why
  - Dense captures semantics; lexical captures exact terms/brands; combining yields robust recall.
  - Different failure modes hedge each other (tail terms vs. semantic drift).
- Trade-offs
  - More moving parts; extra work per query variant.
  - Score calibration differences across signals.
- Mitigations
  - Use rank-based fusion so calibration mismatches matter less.
  - Cap the number of variants and enforce time budgets.
- Metrics to watch
  - Track a Recall@K proxy against single-signal baselines, the latency of the fusion step, and the share of results contributed by each signal.
  - Aggregating heterogeneous signals lowers variance and introduces a slight bias toward consensus items, which reduces idiosyncratic misses of a single modality.
  - The additional retrieval per variant increases compute, but fusion itself is cheap; overall, this typically improves quality per millisecond by reducing tail failures.

---

## 3) Micro query expansion

Decision: A few controlled rewrites (aliases/taxonomy), optionally one low-cost rewrite; no heavy paraphrasing.

- Why
  - Cheap recall boost for synonyms, alias brands, and locale variants without destabilizing meaning.
  - Keeps the query interpretable for explainability.
- Trade-offs
  - Additional retrieval per variant adds latency.
  - Expansion drift can introduce off-target results.
- Mitigations
  - Hard cap on number of expansions and per-variant time budgets.
  - Prefer deterministic expansions; only one low-cost rewrite when strictly beneficial.
- Metrics to watch
  - Measure the expansion hit rate, the incremental recall attributable to expansions, and the rate of expansion-induced errors.
  - Introducing a few controlled expansions adds manageable variance (more hypotheses) while keeping bias aligned to the original intent; adding too many expansions raises variance and noise.
  - Latency grows approximately linearly with the number of expansions but remains bounded by strict caps; by clarifying ambiguous queries, expansions can reduce the need for costly re-ranking, improving throughput.

---

## 4) Two‑stage ranking with a gate

Decision: Use a cheap stage to produce a strong candidate list; re-rank only when it helps.

- Why
  - Most queries don’t need expensive re-ranking; gating keeps latency and cost low while preserving quality.
  - Keeps p50 fast and reserves budget for hard cases.
- Trade-offs
  - Risk of missing improvements when the gate is too conservative.
  - Adds control logic and additional telemetry to tune correctly.
- Mitigations
  - Multi-signal gate using ambiguity and margin heuristics with conservative defaults.
  - Cache re-rank outputs by signature to avoid duplicate work.
- Metrics to watch
  - Observe the distribution of gate modes (SKIP/CHEAP/FULL), the win-rate of CHEAP/FULL against SKIP, and the incremental latency and cost per mode.
  - In SKIP mode the system retains first-stage bias and remains stable; in CHEAP/FULL modes variance increases by exploring a larger hypothesis space, though rubric constraints keep it bounded.
  - Gating yields strong p50 latency improvements by skipping unnecessary stages, while p95 is governed by threshold settings; poorly tuned gates can either waste compute or miss quality gains.

---

## 5) Judge for hard cases, not by default

Decision: Apply a rubric-driven judge only for ambiguous or close-call lists; cache results.

- Why
  - Human-like tie-breaking improves edge cases while keeping default costs minimal.
  - A rubric and cache provide consistency and affordability.
- Trade-offs
  - Additional tokens/time when invoked; potential variability if left unconstrained.
- Mitigations
  - Strict rubric and timeouts; cache keyed by model/rubric/versioned inputs.
  - Budget guardrails to prevent runaway spend.
- Metrics to watch
  - Track the judge invocation rate, the cache hit rate, the quality deltas when the judge is enabled versus skipped, and the token cost per request.
  - The rubric-driven judge reduces variance in close calls by applying a consistent bias; if the rubric is misaligned with user intent, systematic bias can emerge.
  - Invocations increase tail latency and cost, which is mitigated by caching and tight timeouts.

---

## 6) Trust & explainability over cleverness

Decision: Make constraint satisfaction visible (chips), prefer predictable behavior to opaque optimizations.

- Why
  - Users trust systems that explain why items match constraints (budget, color, material, brand).
  - Simpler behavior is easier to audit and debug.
- Trade-offs
  - May leave some potential recall untapped compared to aggressive black-box techniques.
- Mitigations
  - Provide optional detail traces for debugging; keep explanations concise and consistent.
- Metrics to watch
  - Monitor chip coverage, the rate of user corrections (for example, filter toggles after results), and the complaint rate related to violations.
  - Making constraints visible increases both perceived and actual alignment (bias) with user-specified constraints and reduces variance in user expectations.
  - The computational cost is negligible, and clearer diagnostics improve debugging velocity, which indirectly boosts throughput.

---

## 7) Diversity via MMR

Decision: Encourage varied top results and avoid near-duplicates.

- Why
  - Improves browsing experience; reduces redundant items.
- Trade-offs
  - Over-diversification can down-rank the single best cluster too much.
- Mitigations
  - Tune diversity strength conservatively; cap variant groups.
- Metrics to watch
  - Measure the duplicate or near-duplicate rate in the top-N results and monitor downstream engagement metrics.
  - Encouraging diversity intentionally increases variance in the top slate to reduce bias toward a single cluster; overly aggressive settings may under-rank the best cluster.
  - The added compute is small, and better variety can reduce user pogo-sticking, improving perceived responsiveness.

---

## 8) Determinism & reliability over maximum speed tricks

Decision: Favor correctness and stable behavior over aggressive approximations.

- Why
  - Robustness is essential for a take-home demo and for auditability.
- Trade-offs
  - Potentially higher latency than the fastest approximate methods.
- Mitigations
  - Optimize hot paths; parallelize where safe; cache intermediate results.
- Metrics to watch
  - Track p50 and p95 latency by stage, error rates, retries and timeouts, and regression differences across versions.
  - Higher determinism increases bias toward reproducibility and lowers run-to-run variance, which avoids stochastic instability.
  - Although we may sacrifice some raw speed optimizations, stability reduces firefighting and improves overall development velocity.

---

## 9) Observability first

Decision: Expose health and sub-timers for credible tuning; keep traces simple and actionable.

- Why
  - Visibility accelerates iteration and prevents cargo-cult tuning.
- Trade-offs
  - Small implementation overhead.
- Mitigations
  - Minimal, standardized timers and counters; avoid noisy logging.
- Metrics to watch
  - Use the `/healthz` endpoint to confirm index type, size, dimension, readiness, and whether LLM features are enabled; also monitor per-stage timings and gate distributions.
  - While observability does not directly change bias or variance, it enables us to measure shifts and act on them promptly.
  - The runtime overhead is minimal, and the resulting tuning feedback loop offers a large return on investment.

---

## Future Work (if time allowed)

- Learned gate controller: replace heuristic thresholds with a lightweight classifier trained on offline wins/losses; target stable p95 and cost while improving win-rate.
- Fielded lexical upgrade: move from plain TF-style features to BM25F or learned field weights; calibrate with click or curated labels.
- Neural sparse side-signal: add SPLADE/uniCOIL-style sparse embeddings for better long-tail recall without heavy latency.
- Exact vector compression: PCA to 256D while preserving cosine with whitening; evaluate recall and latency deltas; keep exact (non-ANN) path.
- Cross-encoder re-ranker (tiny): introduce a cost-effective learned re-ranker for CHEAP path; keep LLM for FULL (explanations and tie-breaks).
- Personalized priors: session- or cohort-level priors (brand affinity, price sensitivity) as soft weights; guard with constraints first.
- Better expansion policies: bandit selection of expansions under a strict time budget; auto-prune low-yield patterns.
- A/B harness + offline evaluator: fixed query sets with ground-truth proxies; track Recall@K proxy, violation rate, p50/p95, and gate mix.
- Autoscaling + cache warming: warm lexical/vector artifacts and parser caches on deploy; keep cold-start p99 under target.
- Safety & guardrails: budget caps per time window; per-model quotas; robust error stratification in telemetry.

---

## Decision Matrix (summary)

See `docs/design/decision-matrix.csv` for a quick scan of trade-offs across precision, recall, latency, cost, complexity, explainability, and reliability.


