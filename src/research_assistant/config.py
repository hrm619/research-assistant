import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    db_path: str = str(Path.home() / ".research-assistant" / "ra.db")
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_retries: int = 3
    llm_backoff_base: float = 1.0
    llm_backoff_factor: float = 4.0
    log_path: str = str(Path.home() / ".research-assistant" / "ra.log")
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def setup_logging(settings: Settings | None = None) -> None:
    if settings is None:
        settings = get_settings()

    log_dir = os.path.dirname(settings.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(settings.log_path),
            logging.StreamHandler(),
        ],
    )
