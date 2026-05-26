"""Secret-loading utility.

Rules enforced here:
- Secrets come from environment variables (loaded from .env in dev).
- Values are never logged. Even truncated values are not printed.
- A missing secret raises MissingSecretError with the env-var name only.
- The module exposes get_secret() and a Settings model for non-secret config.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root if present. Idempotent.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class MissingSecretError(RuntimeError):
    """Raised when a required secret env var is missing or empty."""


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Fetch a secret from environment.

    Never logs the value. Raises MissingSecretError if required and missing.
    """
    value = os.getenv(name)
    if value is None or value == "":
        if required:
            raise MissingSecretError(
                f"Required secret env var is not set: {name}. "
                f"Add it to your .env (see .env.example)."
            )
        return None
    return value


class AppSettings(BaseSettings):
    """Non-secret runtime settings, loaded from environment.

    Sensitive fields (tokens, API keys) are deliberately NOT here. Use
    get_secret() at the point of use to keep blast radius small.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")
    db_path: str = Field(default="./data/trading.db")
    http_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    )
    http_timeout_seconds: int = Field(default=20)
    http_max_retries: int = Field(default=3)
    cross_source_close_tolerance_pct: float = Field(default=0.10)
    cross_source_volume_tolerance_pct: float = Field(default=2.00)


def get_settings() -> AppSettings:
    return AppSettings()


def project_root() -> Path:
    return _PROJECT_ROOT
