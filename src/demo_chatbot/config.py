"""Central configuration, loaded from .env and validated by pydantic.

Why: one typed, validated source of truth for config. Every other file imports
get_settings() instead of reading os.environ ad-hoc. No stringly-typed bugs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["mock", "ollama", "openai"] = "mock"
    llm_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # Embeddings
    embedder: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    vector_store: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"

    # transport provider API
    db_api_base: str = "https://apis.deutschebahn.com"
    db_api_key: str = ""
    db_client_id: str = ""
    use_real_db_api: bool = False

    # Kafka
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_request_topic: str = "chat.requests"
    kafka_response_topic: str = "chat.responses"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Import this everywhere instead of constructing Settings()."""
    return Settings()