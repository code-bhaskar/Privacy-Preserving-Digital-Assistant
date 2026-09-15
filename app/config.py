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
    # Per-token epsilon for the k-RR release in fl/text_dp.py. Higher keeps more
    # words intact; lower protects more and degrades the escalated prompt.
    escalation_token_epsilon: float = 10.0
    # Which federated stage the learning thread runs. "softmax" federates the
    # full shared intent matrix (fl/pipeline.py); "lora" freezes that matrix and
    # federates a rank-r adapter instead (fl/lora). Both charge the same
    # per-round epsilon/delta to the same lifetime ledger and both publish into
    # model_versions, so switching stages cannot change serving.
    learning_stage: Literal["softmax", "lora"] = "softmax"
    lora_rank: int = 4
    lora_clip_norm: float = 0.1
    lora_local_steps: int = 60
    lora_learning_rate: float = 0.5

    def escalation_epsilon(self):
        if not 0 < self.escalation_token_epsilon <= 20:
            raise RuntimeError("ESCALATION_TOKEN_EPSILON must be in (0, 20]")

    def lora_parameters(self):
        """Validate the adapter stage at startup, not in the middle of a round.

        The clip norm matters for privacy, not just utility: the Gaussian scale
        in fl/privacy.py is calibrated to it, so a non-positive or non-finite
        value would silently break the sensitivity bound. The rank is bounded
        because the released vector is rank * labels long and every coordinate
        is quantised, masked and transmitted.
        """
        import math

        if not 1 <= self.lora_rank <= 32:
            raise RuntimeError("LORA_RANK must be an integer in [1, 32]")
        if not math.isfinite(self.lora_clip_norm) or not 0 < self.lora_clip_norm <= 10:
            raise RuntimeError("LORA_CLIP_NORM must be in (0, 10]")
        if not 1 <= self.lora_local_steps <= 2000:
            raise RuntimeError("LORA_LOCAL_STEPS must be in [1, 2000]")
        if not math.isfinite(self.lora_learning_rate) or not 0 < self.lora_learning_rate <= 4:
            raise RuntimeError("LORA_LEARNING_RATE must be in (0, 4]")

    def keys(self):
        self.escalation_epsilon()
        self.lora_parameters()
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
