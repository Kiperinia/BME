export type PatientGender = '男' | '女' | '其他'

export type PatientExamStatus = 0 | 1 | 2

export interface RawPatientRecord {
  patient_id: string
  patient_name: string
  gender: PatientGender
  age: number
  exam_date: string
  status: PatientExamStatus
}

export interface PatientRecord {
  patientId: string
  patientName: string
  gender: PatientGender
  age: number
  examDate: string
  status: PatientExamStatus
}

export type MaskPoint = [number, number]

export interface PolygonMask {
  id: string
  points: MaskPoint[]
  frameWidth: number
  frameHeight: number
  fillColor?: string
  strokeColor?: string
  needsReview?: boolean
}

export interface VideoFrameData {
  frameId: string
  sourceId: string
  timestamp: number
  width: number
  height: number
  suspectedLocation: string
}

export interface CaptureFramePayload {
  dataUrl: string
  includesMask: boolean
  capturedAt: number
}

export interface AnnotationTag {
  id: string
  label: string
  confidence: number
  targetTime: number
  locationLabel: string
  needsReview: boolean
}

export interface TumorDetails {
  estimatedSizeMm: number
  classification: string
  location: string
  surfacePattern: string
  confidence: number
}

export type TumorMaskData = PolygonMask[] | string

export interface TumorFocusData {
  tumorImageSrc: string
  maskData: TumorMaskData
  details: TumorDetails
}

export interface ReportContextData {
  patient: PatientRecord
  videoSrc: string
  maskData: TumorMaskData
  showMask: boolean
  videoFrameData: VideoFrameData
  captureImageSrcs: string[]
  reportSnippet: string
  initialOpinion: string
  tumorFocus: TumorFocusData
}

export interface GenerateReportDraftRequest {
  reportId?: string
  patientId: string
  contextData: ReportContextData
}

export interface GenerateReportDraftResponse {
  findings: string
  conclusion: string
  layoutSuggestion: string
  workflow: AgentWorkflowSummary
  streamMessages: string[]
}

export interface SaveReportDraftRequest {
  reportId?: string
  patientId: string
  findings: string
  conclusion: string
  layoutSuggestion: string
}

export interface ReportDraftRecord extends SaveReportDraftRequest {
  reportId: string
  updatedAt: string
}

export interface FetchAnnotationTagsRequest {
  contextData: ReportContextData
  reportSnippet: string
}

export interface FetchAnnotationTagsResponse {
  tags: AnnotationTag[]
  workflow: AgentWorkflowSummary
}

export interface AgentWorkflowLesion {
  lesionId: string
  sourceLabel: string
  label: string
  confidence: number
  bbox: [number, number, number, number]
  parisType: string
  invasionRisk: string
  riskLevel: string
  totalScore: number
  disposition: string
  estimatedSizeMm: number
  shapeDescription: string
  usedLlm: boolean
}

export interface AgentToolCall {
  tool_name: string
  status: string
  duration_ms: number
  output_preview: string
  error_message?: string
}

export interface AgentMainTool {
  name: string
  description: string
}

export interface AgentRun {
  agentName: string
  displayName: string
  goal: string
  status: string
  decision: string
  toolCalls: AgentToolCall[]
  observations: Record<string, unknown>
  warnings: string[]
}

export interface AgentDetailSummary {
  agentName: string
  displayName: string
  detail: string
  promptDesign: string[]
  goal: string
  status: string
  decision: string
  mainToolChain: AgentMainTool[]
  warnings: string[]
  keyOutputs: Record<string, unknown>
}

export interface ClosedLoopSummary {
  finalStatus?: string
  finalDecision?: string
  bankDecision?: string
  labelCount?: number
  termCount?: number
  databaseRecordCount?: number
  qualityScore?: number
  agentDetails?: AgentDetailSummary[]
  mainToolChains?: Record<string, AgentMainTool[]>
  warnings?: string[]
}

export interface AgentWorkflowSummary {
  agentName: string
  description: string
  pipeline: string
  llmConfigured: boolean
  workflowMode: string
  generatedAt: string
  lesionCount: number
  highestRiskLesionId?: string | null
  modelVersion: string
  steps: string[]
  warnings: string[]
  lesions: AgentWorkflowLesion[]
  agentRuns?: AgentRun[]
  closedLoopSummary?: ClosedLoopSummary
}

export interface SegmentFrameResponse {
  maskDataUrl: string
  maskCoordinates: MaskPoint[]
  boundingBox: [number, number, number, number]
  maskAreaPixels: number
}

export interface ApiContractDefinition {
  url: string
  method: 'GET' | 'POST' | 'PUT'
  requestType: string
  responseType: string
}
