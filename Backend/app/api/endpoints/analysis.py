from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, UploadFile, status

from app.core.dependencies import get_analysis_service, get_current_user
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.analysis import SubmitTaskRequestSchema, SubmitTaskResponseSchema, TaskStatusResponseSchema
from app.schemas.common import AuthenticatedUserSchema
from app.services.analysis_service import AnalysisService


router = APIRouter()


@router.post(
    "/submit-task",
    response_model=ApiResponse[SubmitTaskResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def submit_task(
    payload: Annotated[SubmitTaskRequestSchema, Depends(SubmitTaskRequestSchema.as_form)],
    image: UploadFile = File(...),
    service: AnalysisService = Depends(get_analysis_service),
    current_user: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[SubmitTaskResponseSchema]:
    try:
        data = await service.submit_task(payload=payload, image=image, current_user=current_user)
        return ApiResponse(data=data)
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50011,
            "failed to submit SAM3 task",
        ) from exc


@router.get(
    "/task-status/{task_id}",
    response_model=ApiResponse[TaskStatusResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def get_task_status(
    task_id: str = Path(..., min_length=36, max_length=36),
    service: AnalysisService = Depends(get_analysis_service),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[TaskStatusResponseSchema]:
    try:
        data = await service.get_task_status(task_id=task_id)
        return ApiResponse(data=data)
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50012,
            "failed to query SAM3 task status",
        ) from exc
