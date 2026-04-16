from __future__ import annotations

from typing import Any
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import UploadFile, status

from app.core.config import Settings
from app.core.enums import TaskStatusEnum
from app.core.exceptions import AppException
from app.models.analysis_task import AnalysisTask
from app.repositories.analysis_task_repository import AnalysisTaskRepository
from app.schemas.analysis import LesionSchema, SubmitTaskRequestSchema, SubmitTaskResponseSchema, TaskStatusResponseSchema
from app.schemas.common import AuthenticatedUserSchema
from app.services.storage_service import StorageService
from app.worker.celery_app import celery_app


class AnalysisService:
    def __init__(
        self,
        repository: AnalysisTaskRepository,
        storage_service: StorageService,
        settings: Settings,
    ):
        self.repository = repository
        self.storage_service = storage_service
        self.settings = settings

    async def submit_task(
        self,
        payload: SubmitTaskRequestSchema,
        image: UploadFile,
        current_user: AuthenticatedUserSchema,
    ) -> SubmitTaskResponseSchema:
        if not current_user.is_authenticated:
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code=40101,
                message="authentication required",
            )

        task_id = str(uuid4())
        image_path = await self.storage_service.persist_upload(task_id=task_id, upload_file=image)

        analysis_task = AnalysisTask(
            task_id=task_id,
            patient_id=payload.patient_id,
            study_id=payload.study_id,
            image_path=image_path,
            requested_by=current_user.user_id,
            status=TaskStatusEnum.PENDING,
        )
        await self.repository.create(analysis_task)

        try:
            celery_app.send_task(
                "app.worker.tasks.run_sam3_segmentation_task",
                args=[task_id, image_path],
                task_id=task_id,
                queue=self.settings.celery_task_queue,
            )
        except Exception as exc:
            await self.repository.update_failure(
                task_id=task_id,
                error_code=50311,
                error_message="failed to enqueue SAM3 task",
            )
            raise AppException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code=50311,
                message="failed to enqueue SAM3 task",
            ) from exc

        return SubmitTaskResponseSchema(task_id=task_id, status=TaskStatusEnum.PENDING)

    async def get_task_status(self, task_id: str) -> TaskStatusResponseSchema:
        task = await self.repository.get_by_task_id(task_id)
        if task is None:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code=40411,
                message="task not found",
            )

        async_result = AsyncResult(task_id, app=celery_app)
        resolved_status = self._resolve_status(task.status, async_result.state)
        payload = task.result_payload or {}

        return TaskStatusResponseSchema(
            task_id=task.task_id,
            status=resolved_status,
            patient_id=task.patient_id,
            study_id=task.study_id,
            submitted_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            mask_coordinates=payload.get("mask_coordinates"),
            lesions=self._build_lesions(task=task, payload=payload),
            error_code=task.error_code,
            error_message=task.error_message,
        )

    @staticmethod
    def _resolve_status(db_status: TaskStatusEnum, celery_state: str) -> TaskStatusEnum:
        if celery_state == "STARTED":
            return TaskStatusEnum.PROCESSING
        if celery_state in TaskStatusEnum._value2member_map_:
            return TaskStatusEnum(celery_state)
        return db_status

    @staticmethod
    def _build_lesions(task: AnalysisTask, payload: dict[str, Any]) -> list[LesionSchema]:
        if task.lesions:
            return [
                LesionSchema(
                    lesion_id=lesion.lesion_id,
                    label=lesion.label,
                    confidence=float(lesion.confidence),
                    location=lesion.location,
                    area_mm2=float(lesion.area_mm2) if lesion.area_mm2 is not None else None,
                    mask_coordinates=lesion.polygon_coordinates,
                )
                for lesion in task.lesions
            ]

        return [LesionSchema(**lesion) for lesion in payload.get("lesions", [])]
