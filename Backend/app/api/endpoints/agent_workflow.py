from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.dependencies import get_current_user, get_sam3_engine
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.agent_workflow import (
    FetchAnnotationTagsRequestSchema,
    FetchAnnotationTagsResponseSchema,
    GenerateReportDraftRequestSchema,
    GenerateReportDraftResponseSchema,
    ReportDraftRecordSchema,
    SaveReportDraftRequestSchema,
)
from app.schemas.common import AuthenticatedUserSchema
from app.services.agent_workflow_service import AgentWorkflowService
from app.services.sam3_runtime import SAM3Engine


router = APIRouter(prefix="/agent")


@router.post(
    "/report-drafts/generate",
    response_model=ApiResponse[GenerateReportDraftResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def generate_report_draft(
    payload: GenerateReportDraftRequestSchema,
    engine: SAM3Engine = Depends(get_sam3_engine),
    settings: Settings = Depends(get_settings),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[GenerateReportDraftResponseSchema]:
    service = AgentWorkflowService(settings=settings, sam3_engine=engine)
    try:
        result = await run_in_threadpool(service.generate_report_draft, payload)
        return ApiResponse(data=GenerateReportDraftResponseSchema(**result))
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50021,
            "failed to run agent report generation workflow",
        ) from exc


@router.post(
    "/annotation-tags/infer",
    response_model=ApiResponse[FetchAnnotationTagsResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def infer_annotation_tags(
    payload: FetchAnnotationTagsRequestSchema,
    engine: SAM3Engine = Depends(get_sam3_engine),
    settings: Settings = Depends(get_settings),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[FetchAnnotationTagsResponseSchema]:
    service = AgentWorkflowService(settings=settings, sam3_engine=engine)
    try:
        result = await run_in_threadpool(service.infer_annotation_tags, payload)
        return ApiResponse(data=FetchAnnotationTagsResponseSchema(**result))
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50022,
            "failed to run agent annotation workflow",
        ) from exc


@router.post(
    "/report-drafts",
    response_model=ApiResponse[ReportDraftRecordSchema],
    status_code=status.HTTP_200_OK,
)
async def save_report_draft(
    payload: SaveReportDraftRequestSchema,
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[ReportDraftRecordSchema]:
    try:
        return ApiResponse(
            data=ReportDraftRecordSchema(
                reportId=payload.reportId or f"draft-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                patientId=payload.patientId,
                findings=payload.findings,
                conclusion=payload.conclusion,
                layoutSuggestion=payload.layoutSuggestion,
                updatedAt=datetime.now(timezone.utc),
            )
        )
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50023,
            "failed to save report draft",
        ) from exc