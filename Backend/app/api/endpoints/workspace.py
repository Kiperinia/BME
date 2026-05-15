from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.dependencies import get_current_user, get_sam3_engine
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.common import AuthenticatedUserSchema
from app.schemas.workspace import (
    ExemplarBankDecisionSchema,
    ExemplarBankRequestSchema,
    WorkspaceReportRequestSchema,
    WorkspaceReportResponseSchema,
)
from app.services.exemplar_bank_service import ExemplarBankService
from app.services.sam3_runtime import SAM3Engine
from app.services.workspace_service import WorkspaceService


router = APIRouter(prefix="/agent/workspace")


@router.post(
    "/report",
    response_model=ApiResponse[WorkspaceReportResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def generate_workspace_report(
    payload: WorkspaceReportRequestSchema,
    engine: SAM3Engine = Depends(get_sam3_engine),
    settings: Settings = Depends(get_settings),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[WorkspaceReportResponseSchema]:
    service = WorkspaceService(settings=settings, sam3_engine=engine)
    try:
        result = await run_in_threadpool(service.generate_report, payload)
        return ApiResponse(data=result)
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50071,
            "failed to generate workspace report",
        ) from exc


@router.post(
    "/exemplar-bank",
    response_model=ApiResponse[ExemplarBankDecisionSchema],
    status_code=status.HTTP_200_OK,
)
async def evaluate_exemplar_candidate(
    payload: ExemplarBankRequestSchema,
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[ExemplarBankDecisionSchema]:
    service = ExemplarBankService()
    try:
        result = await run_in_threadpool(service.evaluate_and_store, payload)
        return ApiResponse(data=result)
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50072,
            "failed to evaluate exemplar-bank candidate",
        ) from exc
