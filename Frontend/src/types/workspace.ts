export interface WorkspacePatient {
  patientId: string
  patientName: string
  examDate: string
}

export interface UploadedWorkspaceImage {
  filename: string
  contentType: string
  dataUrl: string
  objectUrl: string
  width: number
  height: number
  sizeBytes: number
}

export interface WorkspaceSegmentation {
  maskCoordinates: [number, number][]
  boundingBox: [number, number, number, number]
  maskAreaPixels: number
  maskAreaRatio: number
  pointCount: number
}

export interface ExpertConfiguration {
  parisClassification: string
  lesionType: string
  pathologyClassification: string
  surfacePattern: string
  expertNotes: string
}

export interface WorkspaceWorkflowLesion {
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

export interface WorkspaceWorkflowSummary {
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
  lesions: WorkspaceWorkflowLesion[]
}

export interface WorkspaceReportResult {
  findings: string
  conclusion: string
  recommendation: string
  reportMarkdown: string
  workflow: WorkspaceWorkflowSummary
}

export interface ExemplarBankDecision {
  sampleId?: string | null
  accepted: boolean
  score: number
  threshold: number
  reasons: string[]
  duplicateOf?: string | null
  bankSize: number
  storedAt?: string | null
}

export interface WorkspaceReportRequest {
  patient: WorkspacePatient
  image: {
    filename: string
    contentType: string
    dataUrl: string
    width: number
    height: number
  }
  segmentation: WorkspaceSegmentation
  expertConfig: ExpertConfiguration
}

export interface ExemplarBankRequest extends WorkspaceReportRequest {
  reportMarkdown: string
  findings: string
  conclusion: string
}

export interface ToastState {
  visible: boolean
  message: string
  tone: 'info' | 'success' | 'error'
}

export const createDefaultPatient = (): WorkspacePatient => ({
  patientId: 'workspace-case',
  patientName: '',
  examDate: '',
})

export const createDefaultExpertConfiguration = (): ExpertConfiguration => ({
  parisClassification: '',
  lesionType: '',
  pathologyClassification: '',
  surfacePattern: '',
  expertNotes: '',
})
