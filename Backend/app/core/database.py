import logging
from collections.abc import AsyncGenerator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)

_active_backend = "mysql"


def _build_mysql_engine() -> AsyncEngine:
    return create_async_engine(
        settings.mysql_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def _build_sqlite_engine() -> AsyncEngine:
    return create_async_engine(
        settings.sqlite_url,
        echo=settings.debug,
    )

mysql_engine: AsyncEngine = _build_mysql_engine()
sqlite_engine: AsyncEngine = _build_sqlite_engine()

MySQLSessionLocal = async_sessionmaker(
    bind=mysql_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

SQLiteSessionLocal = async_sessionmaker(
    bind=sqlite_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


def get_active_db_backend() -> str:
    return _active_backend


def _switch_to_sqlite(reason: Exception) -> None:
    global _active_backend
    _active_backend = "sqlite"
    logger.warning("MySQL is unavailable, switched database backend to SQLite: %s", reason)


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _active_backend == "sqlite":
        return SQLiteSessionLocal
    return MySQLSessionLocal


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    session_factory = _get_session_factory()
    async with session_factory() as session:
        yield session


async def init_models() -> None:
    from app.models import analysis_task  # noqa: F401
    from app.models import patient  # noqa: F401
    from app.models.base import Base

    global _active_backend

    if settings.db_backend == "sqlite":
        _active_backend = "sqlite"
        async with sqlite_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        return

    _active_backend = "mysql"
    try:
        async with mysql_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except SQLAlchemyError as exc:
        if settings.db_backend == "mysql" or not settings.db_fallback_to_sqlite:
            raise
        _switch_to_sqlite(exc)
        async with sqlite_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)


def get_app_session_factory() -> async_sessionmaker[AsyncSession]:
    return _get_session_factory()
