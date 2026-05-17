from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.dependencies import get_current_user
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.agent_workflow import PatientContextSchema, ReportContextSchema
from app.schemas.common import AuthenticatedUserSchema
from app.services.report_context_service import ReportContextService


router = APIRouter(prefix="/agent")


@router.get(
    "/patient-previews",
    response_model=ApiResponse[list[PatientContextSchema]],
    status_code=status.HTTP_200_OK,
)
async def list_patient_previews(
    session: AsyncSession = Depends(get_db_session),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[list[PatientContextSchema]]:
    service = ReportContextService(session=session)
    try:
        return ApiResponse(data=await service.list_patient_previews())
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50024,
            "failed to load patient preview data",
        ) from exc


@router.get(
    "/report-context",
    response_model=ApiResponse[ReportContextSchema],
    status_code=status.HTTP_200_OK,
)
async def get_report_context(
    report_id: str | None = Query(default=None, alias="reportId"),
    patient_id: str | None = Query(default=None, alias="patientId"),
    session: AsyncSession = Depends(get_db_session),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[ReportContextSchema]:
    service = ReportContextService(session=session)
    try:
        return ApiResponse(data=await service.get_report_context(report_id=report_id, patient_id=patient_id))
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50025,
            "failed to load report context",
        ) from exc