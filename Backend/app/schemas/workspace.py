from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.agent_workflow import AgentWorkflowSchema


class WorkspacePatientSchema(BaseModel):
    patientId: str = Field(default="workspace-patient", min_length=1, max_length=64)
    patientName: str = Field(default="", max_length=128)
    examDate: str = Field(default="", max_length=32)


class WorkspaceImageSchema(BaseModel):
    filename: str = Field(min_length=1, max_length=256)
    contentType: str = Field(default="image/png", max_length=128)
    dataUrl: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class WorkspaceSegmentationSchema(BaseModel):
    maskDataUrl: str = Field(default="")
    maskCoordinates: list[tuple[int, int]] = Field(default_factory=list)
    boundingBox: tuple[int, int, int, int] = Field(default_factory=lambda: (0, 0, 0, 0))
    maskAreaPixels: float = Field(default=0.0, ge=0.0)
    maskAreaRatio: float = Field(default=0.0, ge=0.0, le=1.0)
    pointCount: int = Field(default=0, ge=0)


class ParisDetailSchema(BaseModel):
    morphologyGroup: Literal["elevated", "flat", "depressed"] = "flat"
    selectedSubtypeIndex: int = Field(default=0, ge=0, le=12)
    subtypeCode: str = Field(default="0-IIb", max_length=32)
    displayLabel: str = Field(default="", max_length=64)
    featureSummary: str = Field(default="", max_length=256)
    featureReference: str = Field(default="", max_length=512)


class ExpertConfigurationSchema(BaseModel):
    parisClassification: str = Field(default="", max_length=128)
    parisDetail: ParisDetailSchema = Field(default_factory=ParisDetailSchema)
    lesionType: str = Field(default="", max_length=128)
    pathologyClassification: str = Field(default="", max_length=128)
    surfacePattern: str = Field(default="", max_length=256)
    expertNotes: str = Field(default="", max_length=4000)


class WorkspaceFeatureTagSchema(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    tone: Literal["sky", "emerald", "amber", "rose", "violet"] = "sky"


class AgentTraceStepSchema(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    kind: Literal["thought", "tool_call", "tool_result", "final"] = "thought"
    title: str = Field(min_length=1, max_length=128)
    detail: str = Field(default="", max_length=4000)
    toolName: str | None = Field(default=None, max_length=128)
    status: str | None = Field(default=None, max_length=64)


class WorkspaceReportRequestSchema(BaseModel):
    patient: WorkspacePatientSchema = Field(default_factory=WorkspacePatientSchema)
    image: WorkspaceImageSchema
    segmentation: WorkspaceSegmentationSchema
    expertConfig: ExpertConfigurationSchema = Field(default_factory=ExpertConfigurationSchema)


class WorkspaceReportResponseSchema(BaseModel):
    findings: str
    conclusion: str
    recommendation: str
    reportMarkdown: str
    featureTags: list[WorkspaceFeatureTagSchema] = Field(default_factory=list)
    agentTrace: list[AgentTraceStepSchema] = Field(default_factory=list)
    workflow: AgentWorkflowSchema


class ExemplarBankRequestSchema(BaseModel):
    patient: WorkspacePatientSchema = Field(default_factory=WorkspacePatientSchema)
    image: WorkspaceImageSchema
    segmentation: WorkspaceSegmentationSchema
    expertConfig: ExpertConfigurationSchema = Field(default_factory=ExpertConfigurationSchema)
    reportMarkdown: str = Field(default="", max_length=12000)
    findings: str = Field(default="", max_length=4000)
    conclusion: str = Field(default="", max_length=4000)


class ExemplarBankDecisionSchema(BaseModel):
    sampleId: str | None = None
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    duplicateOf: str | None = None
    bankSize: int = Field(default=0, ge=0)
    storedAt: datetime | None = None
