from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import TaskStatusEnum
from app.models.analysis_task import AnalysisLesion, AnalysisTask


class AnalysisTaskRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, task: AnalysisTask) -> AnalysisTask:
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def get_by_task_id(self, task_id: str) -> AnalysisTask | None:
        statement = (
            select(AnalysisTask)
            .options(selectinload(AnalysisTask.lesions))
            .where(AnalysisTask.task_id == task_id)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def update_processing(self, task_id: str) -> AnalysisTask | None:
        task = await self.get_by_task_id(task_id)
        if task is None:
            return None

        task.status = TaskStatusEnum.PROCESSING
        task.started_at = datetime.now(timezone.utc)
        task.error_code = None
        task.error_message = None
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def update_success(self, task_id: str, payload: dict) -> AnalysisTask | None:
        task = await self.get_by_task_id(task_id)
        if task is None:
            return None

        task.status = TaskStatusEnum.SUCCESS
        task.result_payload = payload
        task.completed_at = datetime.now(timezone.utc)
        task.error_code = None
        task.error_message = None

        task.lesions.clear()
        for lesion in payload.get("lesions", []):
            task.lesions.append(
                AnalysisLesion(
                    label=lesion["label"],
                    location=lesion.get("location"),
                    confidence=Decimal(str(lesion["confidence"])),
                    area_mm2=(
                        Decimal(str(lesion["area_mm2"]))
                        if lesion.get("area_mm2") is not None
                        else None
                    ),
                    polygon_coordinates=lesion["mask_coordinates"],
                    extra_payload={
                        key: value
                        for key, value in lesion.items()
                        if key not in {"label", "location", "confidence", "area_mm2", "mask_coordinates"}
                    }
                    or None,
                )
            )

        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def update_failure(self, task_id: str, error_code: int, error_message: str) -> AnalysisTask | None:
        task = await self.get_by_task_id(task_id)
        if task is None:
            return None

        task.status = TaskStatusEnum.FAILURE
        task.error_code = error_code
        task.error_message = error_message
        task.completed_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(task)
        return task
