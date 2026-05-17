import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.router import api_router
from app.core.config import get_settings
from app.core.database import get_active_db_backend, get_app_session_factory, init_models
from app.core.exceptions import register_exception_handlers
from app.services.report_context_service import ReportContextService
from app.services.sam3_runtime import SAM3RuntimeSingleton


settings = get_settings()
logger = logging.getLogger(__name__)
workspace_dir = Path(__file__).resolve().parents[2]
medex_outputs_dir = workspace_dir / "MedicalSAM3" / "outputs"

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
)

if medex_outputs_dir.exists():
    app.mount(
        "/api/assets/medex-sam3",
        StaticFiles(directory=str(medex_outputs_dir)),
        name="medex-sam3-assets",
    )

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    try:
        await init_models()
        session_factory = get_app_session_factory()
        async with session_factory() as session:
            await _seed_patients(session)
    except Exception as exc:
        logger.warning("Database initialization failed: %s", exc)
    else:
        logger.info("Database backend is ready: %s", get_active_db_backend())
    SAM3RuntimeSingleton.get_instance(settings=settings)


async def _seed_patients(session: AsyncSession) -> None:
    service = ReportContextService(session=session)
    await service.ensure_seed_data()
