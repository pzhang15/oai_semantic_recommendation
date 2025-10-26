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
            self.parser_timeout_secs: int = int(os.getenv("PARSER_TIMEOUT_SECS", "20"))
        except ValueError:
            self.parser_timeout_secs = 20
        try:
            self.parser_max_tokens: int = int(os.getenv("PARSER_MAX_TOKENS", "250"))
        except ValueError:
            self.parser_max_tokens = 250
        try:
            self.parser_retries: int = int(os.getenv("PARSER_RETRIES", "1"))
        except ValueError:
            self.parser_retries = 1

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
            self.mmr_final_k: int = int(os.getenv("MMR_FINAL_K", "0"))
        except ValueError:
            self.mmr_final_k = 0  # 0 means use request limit
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
            self.judge_top_m: int = int(os.getenv("JUDGE_TOP_M", "30"))
        except ValueError:
            self.judge_top_m = 30
        try:
            self.judge_batch_size: int = int(os.getenv("JUDGE_BATCH_SIZE", "8"))
        except ValueError:
            self.judge_batch_size = 8
        try:
            self.judge_timeout_secs: int = int(os.getenv("JUDGE_TIMEOUT_SECS", "12"))
        except ValueError:
            self.judge_timeout_secs = 12
        try:
            self.judge_max_tokens: int = int(os.getenv("JUDGE_MAX_TOKENS", "500"))
        except ValueError:
            self.judge_max_tokens = 500
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


@lru_cache
def get_settings() -> Config:
    return Config()


