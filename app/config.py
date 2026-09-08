import base64
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    database_url: str = "sqlite:///" + str(ROOT / ".runtime/ppda.db")
    jwt_secret: str
    aes_master_key: str
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    cookie_partitioned: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = ""
    ollama_model: str = ""
    pipeline_enabled: bool = True
    session_hours: int = 8
    vapid_private_key_path: str = str(ROOT / ".runtime/vapid-private.pem")
    vapid_subject: str = "mailto:operator@example.com"
    local_summary_model: Literal[
        "flan-t5-small", "t5-small", "distilbart-cnn", "extractive"
    ] = "flan-t5-small"
    local_summary_dir: str = str(ROOT / ".runtime/summary-models")
    snips_dir: str = str(ROOT / "models/snips")

    def keys(self):
        if (
            self.cookie_samesite == "none" or self.cookie_partitioned
        ) and not self.cookie_secure:
            raise RuntimeError(
                "SameSite=None/Partitioned cookies require COOKIE_SECURE=true"
            )
        if len(self.jwt_secret.encode()) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 bytes")
        try:
            key = base64.b64decode(self.aes_master_key, validate=True)
        except Exception as exc:
            raise RuntimeError("AES_MASTER_KEY must be base64") from exc
        if len(key) != 32:
            raise RuntimeError("AES_MASTER_KEY must encode exactly 32 bytes")
        return key


settings = Settings()
MASTER_KEY = settings.keys()
