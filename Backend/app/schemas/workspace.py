from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.agent_workflow import AgentWorkflowSchema


class WorkspacePatientSchema(BaseModel):
    """brief:
        Represent WorkspacePatientSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patientId: str = Field(default="workspace-patient", min_length=1, max_length=64)
    patientName: str = Field(default="", max_length=128)
    examDate: str = Field(default="", max_length=32)


class WorkspaceImageSchema(BaseModel):
    """brief:
        Represent WorkspaceImageSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    filename: str = Field(min_length=1, max_length=256)
    contentType: str = Field(default="image/png", max_length=128)
    dataUrl: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class WorkspaceSegmentationSchema(BaseModel):
    """brief:
        Represent WorkspaceSegmentationSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    maskDataUrl: str = Field(default="")
    maskCoordinates: list[tuple[int, int]] = Field(default_factory=list)
    boundingBox: tuple[int, int, int, int] = Field(default_factory=lambda: (0, 0, 0, 0))
    maskAreaPixels: float = Field(default=0.0, ge=0.0)
    maskAreaRatio: float = Field(default=0.0, ge=0.0, le=1.0)
    pointCount: int = Field(default=0, ge=0)


class ParisDetailSchema(BaseModel):
    """brief:
        Represent ParisDetailSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    morphologyGroup: Literal["elevated", "flat", "depressed"] = "flat"
    selectedSubtypeIndex: int = Field(default=0, ge=0, le=12)
    subtypeCode: str = Field(default="0-IIb", max_length=32)
    displayLabel: str = Field(default="", max_length=64)
    featureSummary: str = Field(default="", max_length=256)
    featureReference: str = Field(default="", max_length=512)


class ExpertConfigurationSchema(BaseModel):
    """brief:
        Represent ExpertConfigurationSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    parisClassification: str = Field(default="", max_length=128)
    parisDetail: ParisDetailSchema = Field(default_factory=ParisDetailSchema)
    lesionType: str = Field(default="", max_length=128)
    pathologyClassification: str = Field(default="", max_length=128)
    surfacePattern: str = Field(default="", max_length=256)
    expertNotes: str = Field(default="", max_length=4000)


class WorkspaceFeatureTagSchema(BaseModel):
    """brief:
        Represent WorkspaceFeatureTagSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    tone: Literal["sky", "emerald", "amber", "rose", "violet"] = "sky"


class AgentTraceStepSchema(BaseModel):
    """brief:
        Represent AgentTraceStepSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    id: str = Field(min_length=1, max_length=128)
    kind: Literal["thought", "tool_call", "tool_result", "final"] = "thought"
    title: str = Field(min_length=1, max_length=128)
    detail: str = Field(default="", max_length=4000)
    toolName: str | None = Field(default=None, max_length=128)
    status: str | None = Field(default=None, max_length=64)


class WorkspaceReportRequestSchema(BaseModel):
    """brief:
        Represent WorkspaceReportRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patient: WorkspacePatientSchema = Field(default_factory=WorkspacePatientSchema)
    image: WorkspaceImageSchema
    segmentation: WorkspaceSegmentationSchema
    expertConfig: ExpertConfigurationSchema = Field(default_factory=ExpertConfigurationSchema)


class WorkspaceReportResponseSchema(BaseModel):
    """brief:
        Represent WorkspaceReportResponseSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    findings: str
    conclusion: str
    recommendation: str
    reportMarkdown: str
    featureTags: list[WorkspaceFeatureTagSchema] = Field(default_factory=list)
    agentTrace: list[AgentTraceStepSchema] = Field(default_factory=list)
    workflow: AgentWorkflowSchema


class ExemplarBankRequestSchema(BaseModel):
    """brief:
        Represent ExemplarBankRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patient: WorkspacePatientSchema = Field(default_factory=WorkspacePatientSchema)
    image: WorkspaceImageSchema
    segmentation: WorkspaceSegmentationSchema
    expertConfig: ExpertConfigurationSchema = Field(default_factory=ExpertConfigurationSchema)
    polarityHint: Literal["positive", "negative", "boundary"] = "positive"
    reportMarkdown: str = Field(default="", max_length=12000)
    findings: str = Field(default="", max_length=4000)
    conclusion: str = Field(default="", max_length=4000)


class ExemplarBankDecisionSchema(BaseModel):
    """brief:
        Represent ExemplarBankDecisionSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    sampleId: str | None = None
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    duplicateOf: str | None = None
    bankSize: int = Field(default=0, ge=0)
    storedAt: datetime | None = None
    bankId: str = Field(default="default-bank", max_length=128)
    memoryState: str | None = Field(default=None, max_length=64)
    qualityBreakdown: dict[str, float] = Field(default_factory=dict)


class ExemplarRetrievalRequestSchema(BaseModel):
    """brief:
        Represent ExemplarRetrievalRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patient: WorkspacePatientSchema = Field(default_factory=WorkspacePatientSchema)
    image: WorkspaceImageSchema
    segmentation: WorkspaceSegmentationSchema
    expertConfig: ExpertConfigurationSchema = Field(default_factory=ExpertConfigurationSchema)
    topK: int = Field(default=6, ge=1, le=32)
    bankId: str = Field(default="default-bank", max_length=128)


class ExemplarRetrievalCandidateSchema(BaseModel):
    """brief:
        Represent ExemplarRetrievalCandidateSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    exemplarId: str = Field(min_length=1, max_length=128)
    polarity: Literal["positive", "negative", "boundary"]
    similarity: float
    rankScore: float
    uncertaintyPenalty: float
    tags: list[str] = Field(default_factory=list)


class ExemplarRetrievalResponseSchema(BaseModel):
    """brief:
        Represent ExemplarRetrievalResponseSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    bankId: str = Field(default="default-bank", max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    promptTokenShape: tuple[int, ...] = Field(default_factory=tuple)
    priorKeys: list[str] = Field(default_factory=list)
    candidateCount: int = Field(default=0, ge=0)
    candidates: list[ExemplarRetrievalCandidateSchema] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)


class ExemplarFeedbackRequestSchema(BaseModel):
    """brief:
        Represent ExemplarFeedbackRequestSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    exemplarId: str = Field(min_length=1, max_length=128)
    bankId: str = Field(default="default-bank", max_length=128)
    failureMode: Literal["false_positive", "false_negative", "uncertain", "success"] = "success"
    qualityScore: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ExemplarFeedbackResponseSchema(BaseModel):
    """brief:
        Represent ExemplarFeedbackResponseSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    exemplarId: str = Field(min_length=1, max_length=128)
    bankId: str = Field(default="default-bank", max_length=128)
    updatedState: str = Field(min_length=1, max_length=64)
    reasons: list[str] = Field(default_factory=list)
    qualityBreakdown: dict[str, float] = Field(default_factory=dict)
