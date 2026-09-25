"""Single source of truth for every env-configurable value.

Replaces scattered `os.getenv(...)` calls across storage/strategies/
benchmark modules. Loaded once via `get_settings()`; reads `.env` itself
(no separate `load_dotenv()` needed), so `iee serve`'s long-running process
picks up config from `.env` in the working directory without the caller
having to `source .env` first.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / Bifrost gateway
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    llm_model: str = "gpt-5.4-mini"
    llm_pricing_json: Optional[str] = None
    embedding_model: str = "openai/text-embedding-3-small"
    # TypeSafe AI's Jev model (docs.typesafe.ai) -- a fast, type-safe
    # structured-decision model, used as an optional pre-classification
    # step ahead of STRUCTURED_EDIT's own escalate detection. None (unset)
    # means the feature is simply off: every caller degrades to the
    # existing pipeline exactly as it behaved before Jev existed.
    typesafe_api_key: Optional[str] = None
    jev_confidence_threshold: float = 0.80
    # Safety cap, not a normal-case limit: a single structured edit or
    # repair should never legitimately need more than this many output
    # tokens. Bounds worst-case cost from a runaway/verbose response
    # without constraining any observed real request.
    max_output_tokens: int = 4000
    # Another safety cap, not a normal-case limit: real generation calls
    # observed throughout this project run well under a few seconds. The
    # OpenAI client's own default (~600s, plus retries) turns a slow or
    # half-broken gateway into a request that hangs for minutes with no
    # feedback instead of failing visibly -- this bounds that without
    # constraining any real, working request.
    llm_request_timeout_seconds: float = 60.0

    # Storage -- MinIO if minio_endpoint is set, local disk otherwise
    minio_endpoint: Optional[str] = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "incremental-poc"
    local_storage_dir: str = "./minio_local_data/incremental-poc"


@lru_cache
def get_settings() -> Settings:
    return Settings()
