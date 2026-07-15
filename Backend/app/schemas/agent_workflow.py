from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PatientContextSchema(BaseModel):
    """brief:
        Represent PatientContextSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patientId: str = Field(min_length=1, max_length=64)
    patientName: str = Field(min_length=1, max_length=128)
    gender: str = Field(min_length=1, max_length=16)
    age: int = Field(ge=0, le=150)
    examDate: str = Field(default="")
    status: int = Field(ge=0, le=2)


class PolygonMaskSchema(BaseModel):
    """brief:
        Represent PolygonMaskSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    id: str = Field(default="")
    points: list[tuple[int, int]] = Field(default_factory=list)
    frameWidth: int = Field(ge=1)
    frameHeight: int = Field(ge=1)
    fillColor: str | None = None
    strokeColor: str | None = None
    needsReview: bool | None = None


class VideoFrameDataSchema(BaseModel):
    """brief:
        Represent VideoFrameDataSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    frameId: str = Field(min_length=1, max_length=128)
    sourceId: str = Field(min_length=1, max_length=128)
    timestamp: float = Field(ge=0.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    suspectedLocation: str = Field(default="", max_length=128)


class TumorDetailsSchema(BaseModel):
    """brief:
        Represent TumorDetailsSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    estimatedSizeMm: float = Field(default=0.0, ge=0.0)
    classification: str = Field(default="", max_length=128)
    location: str = Field(default="", max_length=256)
    surfacePattern: str = Field(default="", max_length=256)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TumorFocusSchema(BaseModel):
    """brief:
        Represent TumorFocusSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    tumorImageSrc: str = Field(min_length=1)
    maskData: list[PolygonMaskSchema] | str = Field(default_factory=list)
    details: TumorDetailsSchema


class ReportContextSchema(BaseModel):
    """brief:
        Represent ReportContextSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
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
    """brief:
        Represent GenerateReportDraftRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    reportId: str | None = None
    patientId: str = Field(min_length=1, max_length=64)
    contextData: ReportContextSchema


class FetchAnnotationTagsRequestSchema(BaseModel):
    """brief:
        Represent FetchAnnotationTagsRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    contextData: ReportContextSchema
    reportSnippet: str = Field(default="")


class SaveReportDraftRequestSchema(BaseModel):
    """brief:
        Represent SaveReportDraftRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    reportId: str | None = None
    patientId: str = Field(min_length=1, max_length=64)
    findings: str = Field(default="")
    conclusion: str = Field(default="")
    layoutSuggestion: str = Field(default="")


class AnnotationTagSchema(BaseModel):
    """brief:
        Represent AnnotationTagSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    targetTime: float = Field(ge=0.0)
    locationLabel: str = Field(default="")
    needsReview: bool = False


class AgentWorkflowLesionSchema(BaseModel):
    """brief:
        Represent AgentWorkflowLesionSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
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


class AgentRunSchema(BaseModel):
    """brief:
        Represent AgentRunSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agentName: str
    displayName: str = Field(default="")
    goal: str = Field(default="")
    status: str = Field(default="")
    decision: str = Field(default="")
    toolCalls: list[dict[str, object]] = Field(default_factory=list)
    observations: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AgentWorkflowSchema(BaseModel):
    """brief:
        Represent AgentWorkflowSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
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
    agentRuns: list[AgentRunSchema] = Field(default_factory=list)
    closedLoopSummary: dict[str, object] = Field(default_factory=dict)


class GenerateReportDraftResponseSchema(BaseModel):
    """brief:
        Represent GenerateReportDraftResponseSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    findings: str
    conclusion: str
    layoutSuggestion: str
    workflow: AgentWorkflowSchema
    streamMessages: list[str] = Field(default_factory=list)


class FetchAnnotationTagsResponseSchema(BaseModel):
    """brief:
        Represent FetchAnnotationTagsResponseSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    tags: list[AnnotationTagSchema] = Field(default_factory=list)
    workflow: AgentWorkflowSchema


class ReportDraftRecordSchema(BaseModel):
    """brief:
        Represent ReportDraftRecordSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    reportId: str
    patientId: str
    findings: str
    conclusion: str
    layoutSuggestion: str
    updatedAt: datetime
