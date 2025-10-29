import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self) -> None:
        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "local")

        # OpenAI
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
            )
        self.openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

        # Models and backend
        self.model_embed: str = os.getenv("MODEL_EMBED", "text-embedding-3-small")
        # Parser/Judge models
        self.model_parser: str = os.getenv("MODEL_PARSER", "gpt-4o-mini")
        self.model_judge: str = os.getenv("MODEL_JUDGE", self.model_parser)
        self.backend: str = os.getenv("BACKEND", "numpy")

        # Data
        self.dataset_path: Optional[str] = os.getenv("DATASET_PATH")
        self.index_path: str = os.getenv("INDEX_PATH", "data/index.faiss")
        self.ids_path: str = os.getenv("IDS_PATH", "data/ids.json")
        self.stats_path: str = os.getenv("STATS_PATH", "data/stats.json")
        self.parquet_path: str = os.getenv("PARQUET_PATH", "data/products.parquet")

        # Defaults
        try:
            self.top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "12"))
        except ValueError:
            self.top_k_default = 12

        # Parser config
        try:
            self.parser_timeout_secs: int = int(os.getenv("PARSER_TIMEOUT_SECS", "8"))
        except ValueError:
            self.parser_timeout_secs = 8
        try:
            self.parser_max_tokens: int = int(os.getenv("PARSER_MAX_TOKENS", "150"))
        except ValueError:
            self.parser_max_tokens = 150
        try:
            self.parser_retries: int = int(os.getenv("PARSER_RETRIES", "0"))
        except ValueError:
            self.parser_retries = 0

        # Facets deterministic parser
        self.facets_mode: str = os.getenv("FACETS_MODE", "det_llm").lower()  # off|det|det_llm
        try:
            self.facets_det_conf_threshold: float = float(os.getenv("FACETS_DET_CONF_THRESHOLD", "0.6"))
        except ValueError:
            self.facets_det_conf_threshold = 0.6

        # Search pipeline
        try:
            self.k_retrieve: int = int(os.getenv("K_RETRIEVE", "400"))
        except ValueError:
            self.k_retrieve = 400
        try:
            self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        except ValueError:
            self.mmr_lambda = 0.7
        try:
            self.mmr_final_k: int = int(os.getenv("MMR_FINAL_K", "200"))
        except ValueError:
            self.mmr_final_k = 200  # 0 means use request limit
        try:
            self.mmr_candidates_max: int = int(os.getenv("MMR_CANDIDATES_MAX", "120"))
        except ValueError:
            self.mmr_candidates_max = 120
        self.mmr_similarity_mode: str = os.getenv("MMR_SIMILARITY_MODE", "vector_then_title")
        try:
            self.mmr_title_sim_threshold: float = float(os.getenv("MMR_TITLE_SIM_THRESHOLD", "0.85"))
        except ValueError:
            self.mmr_title_sim_threshold = 0.85
        # Keys lists: accept CSV
        self.dedup_keys: list[str] = [k.strip() for k in os.getenv("DEDUP_KEYS", "id").split(",") if k.strip()]
        self.variant_keys: list[str] = [k.strip() for k in os.getenv("VARIANT_KEYS", "parent_asin,title_stem,brand").split(",") if k.strip()]
        self.use_judge_default: bool = os.getenv("USE_JUDGE_DEFAULT", "true").lower() == "true"
        try:
            self.judge_top_m: int = int(os.getenv("JUDGE_TOP_M", "20"))
        except ValueError:
            self.judge_top_m = 20
        try:
            self.judge_batch_size: int = int(os.getenv("JUDGE_BATCH_SIZE", "8"))
        except ValueError:
            self.judge_batch_size = 8
        try:
            self.judge_timeout_secs: int = int(os.getenv("JUDGE_TIMEOUT_SECS", "10"))
        except ValueError:
            self.judge_timeout_secs = 10
        try:
            self.judge_max_tokens: int = int(os.getenv("JUDGE_MAX_TOKENS", "220"))
        except ValueError:
            self.judge_max_tokens = 220
        try:
            self.judge_alpha: float = float(os.getenv("JUDGE_ALPHA", "0.5"))
        except ValueError:
            self.judge_alpha = 0.5
        try:
            self.judge_cache_size: int = int(os.getenv("JUDGE_CACHE_SIZE", "4096"))
        except ValueError:
            self.judge_cache_size = 4096
        try:
            self.judge_cache_ttl_secs: int = int(os.getenv("JUDGE_CACHE_TTL_SECS", "86400"))
        except ValueError:
            self.judge_cache_ttl_secs = 86400
        try:
            self.judge_price_tolerance: float = float(os.getenv("JUDGE_PRICE_TOLERANCE", "0.15"))
        except ValueError:
            self.judge_price_tolerance = 0.15
        self.judge_rubric_version: str = os.getenv("JUDGE_RUBRIC_VERSION", "v1")
        try:
            self.judge_concurrency: int = int(os.getenv("JUDGE_CONCURRENCY", "3"))
        except ValueError:
            self.judge_concurrency = 3
        # Judge circuit breaker
        try:
            self.judge_circuit_cooldown_secs: int = int(os.getenv("JUDGE_CIRCUIT_COOLDOWN_SECS", "120"))
        except ValueError:
            self.judge_circuit_cooldown_secs = 120

        # Judge gate (gating, cache, guardrails)
        self.judge_gate_enable: bool = os.getenv("JUDGE_GATE_ENABLE", "true").lower() == "true"
        try:
            self.judge_cheap_top_m: int = int(os.getenv("JUDGE_CHEAP_TOP_M", "10"))
        except ValueError:
            self.judge_cheap_top_m = 10
        try:
            self.judge_full_top_m: int = int(os.getenv("JUDGE_FULL_TOP_M", "20"))
        except ValueError:
            self.judge_full_top_m = 20
        try:
            self.judge_margin_skip_min: float = float(os.getenv("JUDGE_MARGIN_SKIP_MIN", "0.05"))
        except ValueError:
            self.judge_margin_skip_min = 0.05
        try:
            self.judge_entropy_skip_max: float = float(os.getenv("JUDGE_ENTROPY_SKIP_MAX", "1.2"))
        except ValueError:
            self.judge_entropy_skip_max = 1.2
        try:
            self.judge_entropy_tau: float = float(os.getenv("JUDGE_ENTROPY_TAU", "0.15"))
        except ValueError:
            self.judge_entropy_tau = 0.15
        try:
            self.judge_jaccard_skip_min: float = float(os.getenv("JUDGE_JACCARD_SKIP_MIN", "0.6"))
        except ValueError:
            self.judge_jaccard_skip_min = 0.6
        try:
            self.judge_budget_ok_skip_min: float = float(os.getenv("JUDGE_BUDGET_OK_SKIP_MIN", "0.8"))
        except ValueError:
            self.judge_budget_ok_skip_min = 0.8
        try:
            self.judge_category_ok_skip_min: float = float(os.getenv("JUDGE_CATEGORY_OK_SKIP_MIN", "0.7"))
        except ValueError:
            self.judge_category_ok_skip_min = 0.7
        try:
            self.judge_dup_skip_max: float = float(os.getenv("JUDGE_DUP_SKIP_MAX", "0.25"))
        except ValueError:
            self.judge_dup_skip_max = 0.25
        try:
            self.judge_tolerance_budget: float = float(os.getenv("JUDGE_TOLERANCE_BUDGET", "0.05"))
        except ValueError:
            self.judge_tolerance_budget = 0.05
        try:
            self.judge_gate_timeout_secs: int = int(os.getenv("JUDGE_GATE_TIMEOUT_SECS", "10"))
        except ValueError:
            self.judge_gate_timeout_secs = 10
        try:
            self.judge_cost_per_req_usd_max: float = float(os.getenv("JUDGE_COST_PER_REQ_USD_MAX", "0.02"))
        except ValueError:
            self.judge_cost_per_req_usd_max = 0.02
        try:
            self.judge_daily_cost_usd_max: float = float(os.getenv("JUDGE_DAILY_COST_USD_MAX", "10"))
        except ValueError:
            self.judge_daily_cost_usd_max = 10.0
        # Cost sensitivity knobs
        try:
            self.judge_avg_tokens_per_item: int = int(os.getenv("JUDGE_AVG_TOKENS_PER_ITEM", "20"))
        except ValueError:
            self.judge_avg_tokens_per_item = 20
        self.judge_cost_degrade_to_cheap: bool = os.getenv("JUDGE_COST_DEGRADE_TO_CHEAP", "true").lower() == "true"

        # Query expansion
        self.expansion_enabled: bool = os.getenv("EXPANSION_ENABLED", "true").lower() == "true"
        try:
            self.expansion_max_det: int = int(os.getenv("EXPANSION_MAX_DET", "2"))
        except ValueError:
            self.expansion_max_det = 2
        self.expansion_allow_llm: bool = os.getenv("EXPANSION_ALLOW_LLM", "false").lower() == "true"
        try:
            self.expansion_llm_max: int = int(os.getenv("EXPANSION_LLM_MAX", "1"))
        except ValueError:
            self.expansion_llm_max = 1
        try:
            self.expansion_weight_base: float = float(os.getenv("EXPANSION_WEIGHT_BASE", "1.0"))
        except ValueError:
            self.expansion_weight_base = 1.0
        try:
            self.expansion_weight_det1: float = float(os.getenv("EXPANSION_WEIGHT_DET1", "0.8"))
        except ValueError:
            self.expansion_weight_det1 = 0.8
        try:
            self.expansion_weight_det2: float = float(os.getenv("EXPANSION_WEIGHT_DET2", "0.6"))
        except ValueError:
            self.expansion_weight_det2 = 0.6
        try:
            self.expansion_weight_llm: float = float(os.getenv("EXPANSION_WEIGHT_LLM", "0.5"))
        except ValueError:
            self.expansion_weight_llm = 0.5
        try:
            self.rrf_k: int = int(os.getenv("RRF_K", "60"))
        except ValueError:
            self.rrf_k = 60
        try:
            self.expansion_time_budget_ms: float = float(os.getenv("EXPANSION_TIME_BUDGET_MS", "40"))
        except ValueError:
            self.expansion_time_budget_ms = 40.0
        # When FAISS time exceeds budget, still run lexical-only expansions
        self.expansion_lex_only_over_budget: bool = os.getenv("EXPANSION_LEX_ONLY_OVER_BUDGET", "true").lower() == "true"
        # Prefer lex-only expansions to avoid extra embedding calls
        self.expansion_dense_for_det: bool = os.getenv("EXPANSION_DENSE_FOR_DET", "false").lower() == "true"
        # Rubric version bumping for cache invalidation
        try:
            self.rubric_version: int = int(os.getenv("RUBRIC_VERSION", "1"))
        except ValueError:
            self.rubric_version = 1

        # Auto-judge policy
        self.judge_auto_enable: bool = os.getenv("JUDGE_AUTO_ENABLE", "true").lower() == "true"
        self.judge_auto_page1_only: bool = os.getenv("JUDGE_AUTO_PAGE1_ONLY", "true").lower() == "true"
        try:
            self.judge_auto_min_signals: int = int(os.getenv("JUDGE_AUTO_MIN_SIGNALS", "1"))
        except ValueError:
            self.judge_auto_min_signals = 1

        # Outfit composer
        self.outfit_slots_default: list[str] = [
            s.strip() for s in os.getenv("OUTFIT_SLOTS_DEFAULT", "top,bottom,shoes,outerwear,accessories").split(",") if s.strip()
        ]
        self.outfit_required_slots: list[str] = [
            s.strip() for s in os.getenv("OUTFIT_REQUIRED_SLOTS", "top,bottom,shoes").split(",") if s.strip()
        ]
        try:
            self.outfit_max_alternates: int = int(os.getenv("OUTFIT_MAX_ALTERNATES", "2"))
        except ValueError:
            self.outfit_max_alternates = 2
        try:
            self.outfit_candidates_pool: int = int(os.getenv("OUTFIT_CANDIDATES_POOL", "60"))
        except ValueError:
            self.outfit_candidates_pool = 60
        self.outfit_model: str = os.getenv("OUTFIT_MODEL", self.model_parser)
        try:
            self.outfit_timeout_secs: int = int(os.getenv("OUTFIT_TIMEOUT_SECS", "12"))
        except ValueError:
            self.outfit_timeout_secs = 12
        try:
            self.outfit_max_tokens: int = int(os.getenv("OUTFIT_MAX_TOKENS", "700"))
        except ValueError:
            self.outfit_max_tokens = 700
        try:
            self.outfit_retries: int = int(os.getenv("OUTFIT_RETRIES", "1"))
        except ValueError:
            self.outfit_retries = 1
        try:
            self.outfit_price_tolerance: float = float(os.getenv("OUTFIT_PRICE_TOLERANCE", "0.10"))
        except ValueError:
            self.outfit_price_tolerance = 0.10
        self.outfit_color_palette_required: bool = os.getenv("OUTFIT_COLOR_PALETTE_REQUIRED", "true").lower() == "true"

        # Eval harness
        try:
            self.eval_timeout_secs: int = int(os.getenv("EVAL_TIMEOUT_SECS", "30"))
        except ValueError:
            self.eval_timeout_secs = 30
        try:
            self.eval_default_limit: int = int(os.getenv("EVAL_DEFAULT_LIMIT", "12"))
        except ValueError:
            self.eval_default_limit = 12
        try:
            self.eval_qps: float = float(os.getenv("EVAL_QPS", "2"))
        except ValueError:
            self.eval_qps = 2.0
        try:
            self.price_tolerance: float = float(os.getenv("PRICE_TOLERANCE", "0.15"))
        except ValueError:
            self.price_tolerance = 0.15

        # Search quality controls
        self.search_require_price_or_image: bool = os.getenv("SEARCH_REQUIRE_PRICE_OR_IMAGE", "true").lower() == "true"

        # CORS / Frontend origin support
        self.vite_api_base_url: Optional[str] = os.getenv("VITE_API_BASE_URL")
        self.allowed_origins: List[str] = []
        if self.vite_api_base_url:
            self.allowed_origins.append(self.vite_api_base_url)
        extra = os.getenv("ALLOWED_ORIGINS")
        if extra:
            self.allowed_origins.extend([
                origin.strip() for origin in extra.split(",") if origin.strip()
            ])

        # Hybrid retrieval (TF-IDF + Dense + RRF)
        self.hybrid_enabled: bool = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
        try:
            self.hybrid_dense_k: int = int(os.getenv("HYBRID_DENSE_K", "200"))
        except ValueError:
            self.hybrid_dense_k = 200
        try:
            self.hybrid_lex_k: int = int(os.getenv("HYBRID_LEX_K", "200"))
        except ValueError:
            self.hybrid_lex_k = 200
        try:
            self.hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", "75"))
        except ValueError:
            self.hybrid_rrf_k = 75

        # Lexical TF-IDF settings
        try:
            self.lexical_min_df: int = int(os.getenv("LEXICAL_MIN_DF", "2"))
        except ValueError:
            self.lexical_min_df = 2
        try:
            max_df_env = os.getenv("LEXICAL_MAX_DF", "0.98")
            self.lexical_max_df: float = float(max_df_env)
        except ValueError:
            self.lexical_max_df = 0.98
        try:
            ngrams_env = os.getenv("LEXICAL_NGRAMS", "1,2")
            parts = [int(x.strip()) for x in ngrams_env.split(",") if x.strip()]
            if len(parts) == 1:
                parts = [parts[0], parts[0]]
            self.lexical_ngrams: tuple[int, int] = (parts[0], parts[1])
        except Exception:
            self.lexical_ngrams = (1, 2)
        fields_env = os.getenv("LEXICAL_FIELDS", "title,brand,categories,features")
        self.lexical_fields: list[str] = [s.strip() for s in fields_env.split(",") if s.strip()]
        try:
            weights_env = os.getenv("LEXICAL_WEIGHTS", "0.55,0.25,0.10,0.10")
            self.lexical_weights: list[float] = [float(x.strip()) for x in weights_env.split(",") if x.strip()]
        except Exception:
            self.lexical_weights = [0.55, 0.25, 0.10, 0.10]
        # artifact paths
        self.tfidf_path: str = os.getenv("TFIDF_PATH", "data/tfidf_doc.npz")
        self.tfidf_vec_path: str = os.getenv("TFIDF_VEC_PATH", "data/tfidf_vectorizer.joblib")
        self.tfidf_ids_path: str = os.getenv("TFIDF_IDS_PATH", "data/tfidf_ids.json")
        self.tfidf_meta_path: str = os.getenv("TFIDF_META_PATH", "data/tfidf_meta.json")

        # Result-set cache for stable pagination
        self.result_cache_enable: bool = os.getenv("RESULT_CACHE_ENABLE", "true").lower() == "true"
        try:
            self.result_cache_ttl_secs: int = int(os.getenv("RESULT_CACHE_TTL_SECS", "900"))
        except ValueError:
            self.result_cache_ttl_secs = 900
        try:
            self.result_cache_size: int = int(os.getenv("RESULT_CACHE_SIZE", "256"))
        except ValueError:
            self.result_cache_size = 256


@lru_cache
def get_settings() -> Config:
    return Config()


