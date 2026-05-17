import asyncio
import logging
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.dependencies import get_current_user, get_sam3_engine
from app.core.exceptions import AppException, build_http_exception
from app.core.response import ApiResponse
from app.schemas.analysis import Sam3PreloadStatusSchema, SegmentFrameResponseSchema
from app.schemas.common import AuthenticatedUserSchema
from app.services.sam3_runtime import SAM3Engine, SAM3RuntimeSingleton
from app.services.storage_service import StorageService


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/preload-model",
    response_model=ApiResponse[Sam3PreloadStatusSchema],
    status_code=status.HTTP_200_OK,
)
async def preload_sam3_model(
    settings: Settings = Depends(get_settings),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[Sam3PreloadStatusSchema]:
    status_payload = SAM3RuntimeSingleton.ensure_preload_started(settings=settings)
    return ApiResponse(data=Sam3PreloadStatusSchema(**status_payload))


@router.get(
    "/preload-model-status",
    response_model=ApiResponse[Sam3PreloadStatusSchema],
    status_code=status.HTTP_200_OK,
)
async def preload_sam3_model_status(
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[Sam3PreloadStatusSchema]:
    status_payload = SAM3RuntimeSingleton.get_preload_status()
    return ApiResponse(data=Sam3PreloadStatusSchema(**status_payload))


def _validate_segment_upload(image: UploadFile, settings: Settings) -> None:
    filename = image.filename or "upload.bin"
    suffix = FilePath(filename).suffix.lower()
    if suffix not in StorageService.allowed_suffixes:
        raise build_http_exception(
            status.HTTP_400_BAD_REQUEST,
            40011,
            "unsupported image format",
        )

    if image.content_type and not image.content_type.startswith("image/"):
        raise build_http_exception(
            status.HTTP_400_BAD_REQUEST,
            40012,
            "invalid content type for image upload",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if image.size is not None and image.size > max_bytes:
        raise build_http_exception(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            41311,
            "uploaded image exceeds size limit",
        )


@router.post(
    "/segment-frame",
    response_model=ApiResponse[SegmentFrameResponseSchema],
    status_code=status.HTTP_200_OK,
)
async def segment_frame(
    image: UploadFile = File(...),
    engine: SAM3Engine = Depends(get_sam3_engine),
    settings: Settings = Depends(get_settings),
    _: AuthenticatedUserSchema = Depends(get_current_user),
) -> ApiResponse[SegmentFrameResponseSchema]:
    _validate_segment_upload(image=image, settings=settings)

    try:
        image_bytes = await image.read()
    finally:
        await image.close()

    if not image_bytes:
        raise build_http_exception(
            status.HTTP_400_BAD_REQUEST,
            40013,
            "empty image payload",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise build_http_exception(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            41311,
            "uploaded image exceeds size limit",
        )

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(engine.predict_bytes, image_bytes, image.filename),
            timeout=settings.model_inference_timeout_seconds,
        )
        return ApiResponse(data=SegmentFrameResponseSchema(**data))
    except ValueError as exc:
        raise build_http_exception(
            status.HTTP_400_BAD_REQUEST,
            40014,
            str(exc) or "invalid or corrupted image",
        ) from exc
    except asyncio.TimeoutError as exc:
        raise build_http_exception(
            status.HTTP_504_GATEWAY_TIMEOUT,
            50411,
            "SAM3 inference timeout",
        ) from exc
    except AppException as exc:
        raise build_http_exception(exc.status_code, exc.error_code, exc.message) from exc
    except Exception as exc:
        logger.exception(
            "SAM3 segmentation failed for filename=%s content_type=%s size=%s",
            image.filename,
            image.content_type,
            len(image_bytes),
        )
        raise build_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            50013,
            f"failed to segment frame with SAM3: {type(exc).__name__}: {exc}",
        ) from exc
