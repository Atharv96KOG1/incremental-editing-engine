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

    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    llm_model: str = "gpt-5.4-mini"
    llm_pricing_json: Optional[str] = None
    embedding_model: str = "openai/text-embedding-3-small"
    typesafe_api_key: Optional[str] = None
    jev_confidence_threshold: float = 0.80
    max_output_tokens: int = 4000
    llm_request_timeout_seconds: float = 60.0

    minio_endpoint: Optional[str] = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "incremental-poc"
    local_storage_dir: str = "./minio_local_data/incremental-poc"

    redis_url: Optional[str] = None
    pending_confirmation_ttl_seconds: int = 86400


@lru_cache
def get_settings() -> Settings:
    return Settings()
