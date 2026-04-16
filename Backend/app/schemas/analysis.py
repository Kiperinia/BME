from __future__ import annotations

from datetime import datetime

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import TaskStatusEnum


class PointSchema(BaseModel):
    x: int = Field(ge=0, examples=[142])
    y: int = Field(ge=0, examples=[96])


class LesionSchema(BaseModel):
    lesion_id: str | None = Field(default=None, examples=["6a8c5af0-5805-4809-b29c-a0b39ca02fb5"])
    label: str = Field(min_length=2, max_length=64, examples=["suspected_polyp"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.973])
    location: str | None = Field(default=None, max_length=128, examples=["sigmoid_colon"])
    area_mm2: float | None = Field(default=None, ge=0.0, examples=[18.4])
    mask_coordinates: list[PointSchema] = Field(min_length=3)


class SubmitTaskRequestSchema(BaseModel):
    patient_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        examples=["PATIENT_001"],
    )
    study_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        examples=["EXAM_20260416_001"],
    )
    lesion_hint: str | None = Field(default=None, max_length=256, examples=["suspected sigmoid lesion"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "PATIENT_001",
                "study_id": "EXAM_20260416_001",
                "lesion_hint": "suspected sigmoid lesion",
            }
        }
    )

    @classmethod
    def as_form(
        cls,
        patient_id: str = Form(...),
        study_id: str | None = Form(default=None),
        lesion_hint: str | None = Form(default=None),
    ) -> "SubmitTaskRequestSchema":
        return cls(patient_id=patient_id, study_id=study_id, lesion_hint=lesion_hint)


class SubmitTaskResponseSchema(BaseModel):
    task_id: str = Field(examples=["714ce71b-1d25-4a37-b40e-8aa55d4f9744"])
    status: TaskStatusEnum = Field(examples=[TaskStatusEnum.PENDING])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "714ce71b-1d25-4a37-b40e-8aa55d4f9744",
                "status": "PENDING",
            }
        }
    )


class SegmentFrameResponseSchema(BaseModel):
    mask_coordinates: list[tuple[int, int]] = Field(default_factory=list)
    bounding_box: tuple[int, int, int, int] = Field(default_factory=lambda: (0, 0, 0, 0))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mask_coordinates": [[124, 88], [188, 92], [201, 146], [136, 152]],
                "bounding_box": [118, 88, 201, 152],
            }
        }
    )


class TaskStatusResponseSchema(BaseModel):
    task_id: str
    status: TaskStatusEnum
    patient_id: str
    study_id: str | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    mask_coordinates: list[list[PointSchema]] | None = None
    lesions: list[LesionSchema] = Field(default_factory=list)
    error_code: int | None = None
    error_message: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "714ce71b-1d25-4a37-b40e-8aa55d4f9744",
                "status": "SUCCESS",
                "patient_id": "PATIENT_001",
                "study_id": "EXAM_20260416_001",
                "submitted_at": "2026-04-16T13:40:00+08:00",
                "started_at": "2026-04-16T13:40:01+08:00",
                "completed_at": "2026-04-16T13:40:04+08:00",
                "mask_coordinates": [
                    [
                        {"x": 124, "y": 88},
                        {"x": 188, "y": 92},
                        {"x": 201, "y": 146},
                        {"x": 136, "y": 152}
                    ]
                ],
                "lesions": [
                    {
                        "lesion_id": "6a8c5af0-5805-4809-b29c-a0b39ca02fb5",
                        "label": "suspected_polyp",
                        "confidence": 0.973,
                        "location": "sigmoid_colon",
                        "area_mm2": 18.4,
                        "mask_coordinates": [
                            {"x": 124, "y": 88},
                            {"x": 188, "y": 92},
                            {"x": 201, "y": 146},
                            {"x": 136, "y": 152}
                        ]
                    }
                ],
                "error_code": None,
                "error_message": None,
            }
        }
    )
