from pathlib import Path

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.database import init_models
from app.core.exceptions import register_exception_handlers
from app.services.sam3_runtime import SAM3RuntimeSingleton


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    await init_models()
    SAM3RuntimeSingleton.get_instance(settings=settings)
