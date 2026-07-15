from __future__ import annotations

import asyncio
import logging

from app.core.database import AsyncSessionLocal
from app.core.enums import TaskStatusEnum
from app.repositories.analysis_task_repository import AnalysisTaskRepository
from app.services.sam3_runtime import SAM3RuntimeSingleton
from app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)


async def _mark_processing(task_id: str) -> None:
    """brief:
        Handle mark processing.

    parameter:
        - task_id: Input value for task_id.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    async with AsyncSessionLocal() as session:
        repository = AnalysisTaskRepository(session=session)
        await repository.update_processing(task_id=task_id)


async def _mark_success(task_id: str, payload: dict) -> None:
    """brief:
        Handle mark success.

    parameter:
        - task_id: Input value for task_id.
        - payload: Input value for payload.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    async with AsyncSessionLocal() as session:
        repository = AnalysisTaskRepository(session=session)
        await repository.update_success(task_id=task_id, payload=payload)


async def _mark_failure(task_id: str, error_code: int, error_message: str) -> None:
    """brief:
        Handle mark failure.

    parameter:
        - task_id: Input value for task_id.
        - error_code: Input value for error_code.
        - error_message: Input value for error_message.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    async with AsyncSessionLocal() as session:
        repository = AnalysisTaskRepository(session=session)
        await repository.update_failure(
            task_id=task_id,
            error_code=error_code,
            error_message=error_message,
        )


@celery_app.task(bind=True, name="app.worker.tasks.run_sam3_segmentation_task")
def run_sam3_segmentation_task(self, task_id: str, image_path: str) -> dict:
    """brief:
        Run sam3 segmentation task.

    parameter:
        - task_id: Input value for task_id.
        - image_path: Input value for image_path.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    logger.info("SAM3 task started: %s", task_id)
    asyncio.run(_mark_processing(task_id=task_id))
    self.update_state(
        state=TaskStatusEnum.PROCESSING.value,
        meta={"task_id": task_id, "status": TaskStatusEnum.PROCESSING.value},
    )

    try:
        runtime = SAM3RuntimeSingleton.get_instance()
        result = runtime.run_inference(image_path=image_path)
        asyncio.run(_mark_success(task_id=task_id, payload=result))
        logger.info("SAM3 task finished successfully: %s", task_id)
        return {
            "task_id": task_id,
            "status": TaskStatusEnum.SUCCESS.value,
            **result,
        }
    except Exception as exc:
        error_message = str(exc)
        logger.exception("SAM3 task failed: %s", task_id)
        asyncio.run(_mark_failure(task_id=task_id, error_code=52011, error_message=error_message))
        self.update_state(
            state=TaskStatusEnum.FAILURE.value,
            meta={
                "task_id": task_id,
                "status": TaskStatusEnum.FAILURE.value,
                "error_code": 52011,
                "error_message": error_message,
            },
        )
        raise
