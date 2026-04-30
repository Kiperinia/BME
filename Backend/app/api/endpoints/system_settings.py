from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_current_user
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.common import AuthenticatedUserSchema
from app.schemas.system_settings import SystemSettingsPayloadSchema, SystemSettingsResponseSchema
from app.services.system_settings_service import SystemSettingsService


router = APIRouter(prefix="/system")


@router.get(
    "/settings",
    response_model=ApiResponse[SystemSettingsResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def get_system_settings(
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[SystemSettingsResponseSchema]:
    service = SystemSettingsService()
    try:
        return ApiResponse(data=service.get_system_settings())
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50041,
            "failed to load system settings",
        ) from exc


@router.put(
    "/settings",
    response_model=ApiResponse[SystemSettingsResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def update_system_settings(
    payload: SystemSettingsPayloadSchema,
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[SystemSettingsResponseSchema]:
    service = SystemSettingsService()
    try:
        return ApiResponse(data=service.update_system_settings(payload))
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50042,
            "failed to update system settings",
        ) from exc