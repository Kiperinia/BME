from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PatientContextSchema(BaseModel):
    patientId: str = Field(min_length=1, max_length=64)
    patientName: str = Field(min_length=1, max_length=128)
    gender: str = Field(min_length=1, max_length=16)
    age: int = Field(ge=0, le=150)
    examDate: str = Field(default="")
    status: int = Field(ge=0, le=2)


class PolygonMaskSchema(BaseModel):
    id: str = Field(default="")
    points: list[tuple[int, int]] = Field(default_factory=list)
    frameWidth: int = Field(ge=1)
    frameHeight: int = Field(ge=1)
    fillColor: str | None = None
    strokeColor: str | None = None
    needsReview: bool | None = None


class VideoFrameDataSchema(BaseModel):
    frameId: str = Field(min_length=1, max_length=128)
    sourceId: str = Field(min_length=1, max_length=128)
    timestamp: float = Field(ge=0.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    suspectedLocation: str = Field(default="", max_length=128)


class TumorDetailsSchema(BaseModel):
    estimatedSizeMm: float = Field(default=0.0, ge=0.0)
    classification: str = Field(default="", max_length=128)
    location: str = Field(default="", max_length=256)
    surfacePattern: str = Field(default="", max_length=256)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TumorFocusSchema(BaseModel):
    tumorImageSrc: str = Field(min_length=1)
    maskData: list[PolygonMaskSchema] | str = Field(default_factory=list)
    details: TumorDetailsSchema


class ReportContextSchema(BaseModel):
    patient: PatientContextSchema
    videoSrc: str = Field(default="")
    maskData: list[PolygonMaskSchema] = Field(default_factory=list)
    showMask: bool = True
    videoFrameData: VideoFrameDataSchema
    captureImageSrcs: list[str] = Field(default_factory=list)
    reportSnippet: str = Field(default="")
    initialOpinion: str = Field(default="")
    tumorFocus: TumorFocusSchema


class GenerateReportDraftRequestSchema(BaseModel):
    reportId: str | None = None
    patientId: str = Field(min_length=1, max_length=64)
    contextData: ReportContextSchema


class FetchAnnotationTagsRequestSchema(BaseModel):
    contextData: ReportContextSchema
    reportSnippet: str = Field(default="")


class SaveReportDraftRequestSchema(BaseModel):
    reportId: str | None = None
    patientId: str = Field(min_length=1, max_length=64)
    findings: str = Field(default="")
    conclusion: str = Field(default="")
    layoutSuggestion: str = Field(default="")


class AnnotationTagSchema(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    targetTime: float = Field(ge=0.0)
    locationLabel: str = Field(default="")
    needsReview: bool = False


class AgentWorkflowLesionSchema(BaseModel):
    lesionId: str
    sourceLabel: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int]
    parisType: str
    invasionRisk: str
    riskLevel: str
    totalScore: float = Field(ge=0.0)
    disposition: str
    estimatedSizeMm: float = Field(ge=0.0)
    shapeDescription: str = Field(default="")
    usedLlm: bool = False


class AgentWorkflowSchema(BaseModel):
    agentName: str
    description: str
    pipeline: str
    llmConfigured: bool
    workflowMode: str
    generatedAt: datetime
    lesionCount: int = Field(ge=0)
    highestRiskLesionId: str | None = None
    modelVersion: str = Field(default="")
    steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lesions: list[AgentWorkflowLesionSchema] = Field(default_factory=list)


class GenerateReportDraftResponseSchema(BaseModel):
    findings: str
    conclusion: str
    layoutSuggestion: str
    workflow: AgentWorkflowSchema
    streamMessages: list[str] = Field(default_factory=list)


class FetchAnnotationTagsResponseSchema(BaseModel):
    tags: list[AnnotationTagSchema] = Field(default_factory=list)
    workflow: AgentWorkflowSchema


class ReportDraftRecordSchema(BaseModel):
    reportId: str
    patientId: str
    findings: str
    conclusion: str
    layoutSuggestion: str
    updatedAt: datetime