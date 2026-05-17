import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = BACKEND_DIR.parent
RUNTIME_SETTINGS_PATH = BACKEND_DIR / "runtime" / "system_settings.json"


class Settings(BaseSettings):
    app_name: str = Field(default="BME Async SAM3 Backend")
    debug: bool = Field(default=False)
    api_v1_prefix: str = Field(default="/api")

    mysql_url: str = Field(
        default="mysql+asyncmy://root:password@127.0.0.1:3306/bme",
        alias="MYSQL_URL",
    )
    sqlite_url: str = Field(
        default=f"sqlite+aiosqlite:///{(BACKEND_DIR / 'runtime' / 'bme-dev.sqlite3').resolve().as_posix()}",
        alias="SQLITE_URL",
    )
    db_backend: str = Field(default="auto", alias="DB_BACKEND", pattern="^(auto|mysql|sqlite)$")
    db_fallback_to_sqlite: bool = Field(default=True, alias="DB_FALLBACK_TO_SQLITE")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    celery_broker_url: str | None = Field(default=None, alias="CELERY_BROKER_URL")
    celery_result_backend: str | None = Field(default=None, alias="CELERY_RESULT_BACKEND")
    celery_task_queue: str = Field(default="sam3_analysis", alias="CELERY_TASK_QUEUE")
    celery_worker_concurrency: int = Field(default=1, alias="CELERY_WORKER_CONCURRENCY", ge=1, le=4)
    celery_worker_prefetch_multiplier: int = Field(
        default=1,
        alias="CELERY_WORKER_PREFETCH_MULTIPLIER",
        ge=1,
        le=8,
    )
    celery_worker_pool: str = Field(default="solo", alias="CELERY_WORKER_POOL")
    celery_task_soft_time_limit: int = Field(default=120, alias="CELERY_TASK_SOFT_TIME_LIMIT", ge=30)
    celery_task_time_limit: int = Field(default=180, alias="CELERY_TASK_TIME_LIMIT", ge=60)
    celery_result_expires: int = Field(default=3600, alias="CELERY_RESULT_EXPIRES", ge=60)

    upload_dir: str = Field(
        default=str((BACKEND_DIR / "uploads").resolve()),
        alias="UPLOAD_DIR",
    )
    model_load_mode: str = Field(default="mock", alias="MODEL_LOAD_MODE", pattern="^(mock|sam3)$")
    model_device: str = Field(default="cuda", alias="MODEL_DEVICE")
    model_input_size: int = Field(default=1024, alias="MODEL_INPUT_SIZE", ge=256, le=4096)
    model_checkpoint_path: str = Field(
        default=str((WORKSPACE_DIR / "MedicalSAM3" / "checkpoint" / "MedSAM3.pt").resolve()),
        alias="MODEL_CHECKPOINT_PATH",
    )
    model_lora_enabled: bool = Field(default=False, alias="MODEL_LORA_ENABLED")
    model_lora_path: str = Field(default="", alias="MODEL_LORA_PATH")
    model_lora_stage: str = Field(default="stage_a", alias="MODEL_LORA_STAGE", pattern="^(stage_a|stage_b|stage_c)$")
    model_keep_aspect_ratio: bool = Field(default=False, alias="MODEL_KEEP_ASPECT_RATIO")
    model_warmup_enabled: bool = Field(default=True, alias="MODEL_WARMUP_ENABLED")
    model_inference_timeout_seconds: int = Field(
        default=20,
        alias="MODEL_INFERENCE_TIMEOUT_SECONDS",
        ge=1,
        le=300,
    )
    model_mask_threshold: float = Field(default=0.5, alias="MODEL_MASK_THRESHOLD", ge=0.0, le=1.0)
    model_polygon_epsilon_ratio: float = Field(
        default=0.01,
        alias="MODEL_POLYGON_EPSILON_RATIO",
        gt=0.0,
        lt=1.0,
    )
    model_min_contour_area: int = Field(default=64, alias="MODEL_MIN_CONTOUR_AREA", ge=0)
    model_mock_delay_ms: int = Field(default=0, alias="MODEL_MOCK_DELAY_MS", ge=0, le=10000)
    max_upload_size_mb: int = Field(default=20, alias="MAX_UPLOAD_SIZE_MB", ge=1, le=200)
    auth_header_name: str = Field(default="X-User-Id", alias="AUTH_HEADER_NAME")
    agent_use_llm: bool = Field(default=True, alias="AGENT_USE_LLM")
    agent_use_llm_report: bool = Field(default=True, alias="AGENT_USE_LLM_REPORT")
    agent_pixel_size_mm: float = Field(default=0.15, alias="AGENT_PIXEL_SIZE_MM", gt=0.0, le=10.0)

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


def get_runtime_settings_path() -> Path:
    return RUNTIME_SETTINGS_PATH


def load_settings_overrides() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {}

    raw = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"runtime settings file must be a JSON object: {RUNTIME_SETTINGS_PATH}")
    return raw


def save_settings_overrides(overrides: dict[str, Any]) -> None:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_SETTINGS_PATH.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def refresh_settings_cache() -> None:
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**load_settings_overrides())
