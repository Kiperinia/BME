from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db_session
from app.repositories.analysis_task_repository import AnalysisTaskRepository
from app.schemas.common import AuthenticatedUserSchema
from app.services.analysis_service import AnalysisService
from app.services.sam3_runtime import SAM3Engine, SAM3RuntimeSingleton
from app.services.storage_service import StorageService


class HeaderAuthService:
    """brief:
        Represent HeaderAuthService state and behavior.

    parameter:
        - settings: Input value for settings.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, settings: Settings):
        """brief:
            Initialize this object.

        parameter:
            - settings: Input value for settings.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.header_name = settings.auth_header_name

    async def authenticate(self, request: Request) -> AuthenticatedUserSchema:
        """brief:
            Handle authenticate.

        parameter:
            - request: Input value for request.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        user_id = request.headers.get(self.header_name, "system")
        return AuthenticatedUserSchema(user_id=user_id, is_authenticated=True, role="developer")


def get_auth_service(settings: Settings = Depends(get_settings)) -> HeaderAuthService:
    """brief:
        Get auth service.

    parameter:
        - settings: Input value for settings.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return HeaderAuthService(settings=settings)


async def get_current_user(
    request: Request,
    auth_service: HeaderAuthService = Depends(get_auth_service),
) -> AuthenticatedUserSchema:
    """brief:
        Get current user.

    parameter:
        - request: Input value for request.
        - auth_service: Input value for auth_service.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return await auth_service.authenticate(request)


def get_analysis_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> AnalysisService:
    """brief:
        Get analysis service.

    parameter:
        - session: Input value for session.
        - settings: Input value for settings.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    repository = AnalysisTaskRepository(session=session)
    storage_service = StorageService(settings=settings)
    return AnalysisService(
        repository=repository,
        storage_service=storage_service,
        settings=settings,
    )


def get_sam3_engine(settings: Settings = Depends(get_settings)) -> SAM3Engine:
    """brief:
        Get sam3 engine.

    parameter:
        - settings: Input value for settings.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return SAM3RuntimeSingleton.get_instance(settings=settings).engine
