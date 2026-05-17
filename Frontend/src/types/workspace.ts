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
  maskDataUrl: string
  maskCoordinates: [number, number][]
  boundingBox: [number, number, number, number]
  maskAreaPixels: number
  maskAreaRatio: number
  pointCount: number
  retrievalApplied?: boolean
  retrievalConfidence?: number | null
  retrievalUncertainty?: number | null
  retrievalCandidateCount?: number
  retrievalBankId?: string | null
  retrievalPriorKeys?: string[]
}

export type ParisMorphologyGroup = 'elevated' | 'flat' | 'depressed'
export type FeatureTagTone = 'sky' | 'emerald' | 'amber' | 'rose' | 'violet'

export interface ParisSubtypeOption {
  code: string
  label: string
  summary: string
  featureReference: string
}

export interface DetailedParisConfiguration {
  morphologyGroup: ParisMorphologyGroup
  selectedSubtypeIndex: number
  subtypeCode: string
  displayLabel: string
  featureSummary: string
  featureReference: string
}

export interface ExpertConfiguration {
  parisClassification: string
  parisDetail: DetailedParisConfiguration
  lesionType: string
  pathologyClassification: string
  surfacePattern: string
  expertNotes: string
}

export interface WorkspaceFeatureTag {
  id: string
  label: string
  category: string
  tone: FeatureTagTone
}

export interface AgentTraceStep {
  id: string
  kind: 'thought' | 'tool_call' | 'tool_result' | 'final'
  title: string
  detail: string
  toolName?: string | null
  status?: string | null
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
  featureTags: WorkspaceFeatureTag[]
  agentTrace: AgentTraceStep[]
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
  bankId: string
  memoryState?: string | null
  qualityBreakdown: Record<string, number>
}

export type ExemplarPolarityHint = 'positive' | 'negative' | 'boundary'

export interface ExemplarRetrievalCandidate {
  exemplarId: string
  polarity: ExemplarPolarityHint
  similarity: number
  rankScore: number
  uncertaintyPenalty: number
  tags: string[]
}

export interface ExemplarRetrievalResult {
  bankId: string
  confidence: number
  uncertainty: number
  promptTokenShape: number[]
  priorKeys: string[]
  candidateCount: number
  candidates: ExemplarRetrievalCandidate[]
  diagnostics: Record<string, unknown>
}

export type ExemplarFeedbackMode = 'false_positive' | 'false_negative' | 'uncertain' | 'success'

export interface ExemplarFeedbackResult {
  exemplarId: string
  bankId: string
  updatedState: string
  reasons: string[]
  qualityBreakdown: Record<string, number>
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
  polarityHint: ExemplarPolarityHint
  reportMarkdown: string
  findings: string
  conclusion: string
}

export interface ExemplarRetrievalRequest extends WorkspaceReportRequest {
  topK: number
  bankId: string
}

export interface ExemplarFeedbackRequest {
  exemplarId: string
  bankId: string
  failureMode: ExemplarFeedbackMode
  qualityScore?: number
  uncertainty?: number
  metadata?: Record<string, unknown>
}

export interface PatientCaseRecord {
  recordId: string
  patient: WorkspacePatient
  createdAt: string
  imageFilename: string
  findings: string
  conclusion: string
  recommendation: string
  reportMarkdown: string
  featureTags: WorkspaceFeatureTag[]
  parisClassification: string
  lesionType: string
  pathologyClassification: string
  workflowMode: string
  riskLevel: string
}

export interface ToastState {
  visible: boolean
  message: string
  tone: 'info' | 'success' | 'error'
}

export const PARIS_GROUP_LABELS: Record<ParisMorphologyGroup, string> = {
  elevated: '隆起型',
  flat: '平坦型',
  depressed: '凹陷型',
}

export const PARIS_SUBTYPE_OPTIONS: Record<ParisMorphologyGroup, ParisSubtypeOption[]> = {
  elevated: [
    {
      code: '0-Ip',
      label: '有蒂隆起',
      summary: '头部明显突出，蒂部清晰。',
      featureReference: '病灶向腔内突起明显，通常可见清晰蒂部和相对集中的头端表面结构。',
    },
    {
      code: '0-Isp',
      label: '亚蒂隆起',
      summary: '隆起明显，但蒂部不完全形成。',
      featureReference: '病灶基底较宽，隆起形态突出，连接部呈短蒂或过渡样结构。',
    },
    {
      code: '0-Is',
      label: '无蒂隆起',
      summary: '整体向腔内隆起，基底宽广。',
      featureReference: '病灶为宽基底隆起，边界清楚，整体向腔内抬升但无明显蒂部。',
    },
    {
      code: '0-IIa',
      label: '轻微隆起',
      summary: '轻微隆起，边缘平缓。',
      featureReference: '病灶仅轻度高于周围黏膜，边缘过渡柔和，表面结构常较完整。',
    },
  ],
  flat: [
    {
      code: '0-IIa',
      label: '轻微隆起',
      summary: '轻微隆起，边缘平缓。',
      featureReference: '病灶略高于周围黏膜，整体轮廓平缓，常需结合纹理和血管征象判断。',
    },
    {
      code: '0-IIa+IIb',
      label: '隆起伴平坦过渡',
      summary: '以轻度隆起为主，局部平坦过渡。',
      featureReference: '病灶中央或主体轻度抬高，周缘逐渐过渡到平坦黏膜，边界需仔细辨认。',
    },
    {
      code: '0-IIb',
      label: '完全平坦',
      summary: '完全平坦，与周围黏膜齐平。',
      featureReference: '病灶与周围黏膜基本齐平，常依赖色泽、腺管和血管纹理变化进行识别。',
    },
    {
      code: '0-IIb+IIc',
      label: '平坦伴浅凹陷',
      summary: '平坦背景上伴局灶浅凹陷。',
      featureReference: '大部分区域与黏膜齐平，但局部可见轻度下陷或表面破坏，应注意高风险征象。',
    },
  ],
  depressed: [
    {
      code: '0-IIc',
      label: '轻微凹陷',
      summary: '轻微凹陷，边缘可伴充血或糜烂。',
      featureReference: '病灶表面局部下陷，边缘可能伴发红、糜烂或不规则血管，需重点警惕浸润风险。',
    },
    {
      code: '0-IIa+IIc',
      label: '隆起伴中央凹陷',
      summary: '隆起病灶中央伴浅凹陷。',
      featureReference: '病灶整体抬高，但中央可见浅凹陷或结构破坏，常提示局部高级别异常。',
    },
    {
      code: '0-IIc+IIa',
      label: '凹陷周缘轻度隆起',
      summary: '浅凹陷周缘轻度隆起。',
      featureReference: '病灶中央下陷，周边形成轻度堤状抬高，边缘结构和血管异常应重点记录。',
    },
    {
      code: '0-III',
      label: '明显凹陷',
      summary: '明显凹陷或溃疡样改变。',
      featureReference: '病灶凹陷明显，可能伴溃疡样破坏、充血或覆白苔，应高度警惕深部浸润。',
    },
  ],
}

const buildParisClassification = (detail: DetailedParisConfiguration) => {
  return `${PARIS_GROUP_LABELS[detail.morphologyGroup]} / ${detail.subtypeCode} ${detail.displayLabel}：${detail.featureSummary}`
}

export const createFormalPatientId = () => {
  const now = new Date()
  const yyyy = now.getFullYear()
  const mm = String(now.getMonth() + 1).padStart(2, '0')
  const dd = String(now.getDate()).padStart(2, '0')
  const suffix = Math.floor(Math.random() * 10000).toString().padStart(4, '0')
  return `PAT-${yyyy}${mm}${dd}-${suffix}`
}

export const createDefaultPatient = (): WorkspacePatient => ({
  patientId: createFormalPatientId(),
  patientName: '',
  examDate: '',
})

export const createDefaultParisDetail = (): DetailedParisConfiguration => {
  const fallbackOption: ParisSubtypeOption = {
    code: '0-IIb',
    label: '完全平坦',
    summary: '完全平坦，与周围黏膜齐平。',
    featureReference: '病灶与周围黏膜基本齐平，常依赖色泽、腺管和血管纹理变化进行识别。',
  }
  const defaultOption = PARIS_SUBTYPE_OPTIONS.flat[2] ?? fallbackOption

  return {
    morphologyGroup: 'flat',
    selectedSubtypeIndex: 2,
    subtypeCode: defaultOption.code,
    displayLabel: defaultOption.label,
    featureSummary: defaultOption.summary,
    featureReference: defaultOption.featureReference,
  }
}

export const createDefaultExpertConfiguration = (): ExpertConfiguration => {
  const parisDetail = createDefaultParisDetail()

  return {
    parisClassification: buildParisClassification(parisDetail),
    parisDetail,
    lesionType: '',
    pathologyClassification: '',
    surfacePattern: '',
    expertNotes: '',
  }
}

export const createParisDetailFromSelection = (
  morphologyGroup: ParisMorphologyGroup,
  selectedSubtypeIndex: number,
): DetailedParisConfiguration => {
  const options = PARIS_SUBTYPE_OPTIONS[morphologyGroup]
  const safeIndex = Math.min(Math.max(selectedSubtypeIndex, 0), options.length - 1)
  const fallbackOption: ParisSubtypeOption = {
    code: '0-IIb',
    label: '完全平坦',
    summary: '完全平坦，与周围黏膜齐平。',
    featureReference: '病灶与周围黏膜基本齐平，常依赖色泽、腺管和血管纹理变化进行识别。',
  }
  const option = options[safeIndex] ?? options[0] ?? fallbackOption

  return {
    morphologyGroup,
    selectedSubtypeIndex: safeIndex,
    subtypeCode: option.code,
    displayLabel: option.label,
    featureSummary: option.summary,
    featureReference: option.featureReference,
  }
}

export const formatParisClassification = (detail: DetailedParisConfiguration) => {
  return buildParisClassification(detail)
}
