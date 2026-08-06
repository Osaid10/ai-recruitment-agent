"""Settings, loaded from `.env` / environment. No key ever lives in source."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root = three levels up from this file (src/recruiter/config.py)
ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the project runs even without python-dotenv.

    Existing environment variables always win, which is what you want when
    running in CI or with a key exported in the shell.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    # Groq's free tier meters tokens per minute. Requests are paced to stay
    # under this so stages don't get rate-limited into their fallback paths.
    tokens_per_minute: int = 12_000
    db_path: Path = ROOT / "data" / "ats.db"
    out_dir: Path = ROOT / "out"
    redact_for_ranking: bool = True
    company_name: str = "Mercurial Minds"
    timezone: str = "Asia/Karachi"

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        _load_dotenv(env_file or (ROOT / ".env"))
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            model=os.environ.get("RECRUITER_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.environ.get("RECRUITER_TEMPERATURE", "0.1")),
            tokens_per_minute=int(os.environ.get("RECRUITER_TOKENS_PER_MINUTE", "12000")),
            db_path=_resolve(os.environ.get("RECRUITER_DB_PATH", "data/ats.db")),
            out_dir=_resolve(os.environ.get("RECRUITER_OUT_DIR", "out")),
            redact_for_ranking=_as_bool(
                os.environ.get("RECRUITER_REDACT_FOR_RANKING", "true"), True
            ),
            company_name=os.environ.get("RECRUITER_COMPANY_NAME", "Mercurial Minds"),
            timezone=os.environ.get("RECRUITER_TIMEZONE", "Asia/Karachi"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key)
