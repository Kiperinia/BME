from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import TaskStatusEnum
from app.models.base import Base


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    study_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[TaskStatusEnum] = mapped_column(
        Enum(TaskStatusEnum, native_enum=False, length=16),
        default=TaskStatusEnum.PENDING,
        index=True,
        nullable=False,
    )
    result_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    lesions: Mapped[list[AnalysisLesion]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class AnalysisLesion(Base):
    __tablename__ = "analysis_lesions"

    lesion_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_tasks.task_id", ondelete="CASCADE"),
        index=True,
    )
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    location: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    area_mm2: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    polygon_coordinates: Mapped[list[dict[str, int]]] = mapped_column(JSON, nullable=False)
    extra_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    task: Mapped[AnalysisTask] = relationship(back_populates="lesions")
