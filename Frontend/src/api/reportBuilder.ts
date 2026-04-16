import axios from 'axios'

import type {
  AnnotationTag,
  ApiContractDefinition,
  FetchAnnotationTagsRequest,
  GenerateReportDraftRequest,
  GenerateReportDraftResponse,
  PatientRecord,
  PolygonMask,
  RawPatientRecord,
  ReportContextData,
  ReportDraftRecord,
  SaveReportDraftRequest,
  TumorDetails,
} from '@/types/eis'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'
const agentApiBaseUrl = import.meta.env.VITE_AGENT_API_BASE_URL ?? '/agent-api'

export const httpClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15000,
})

export const reportBuilderApiContracts: Record<string, ApiContractDefinition> = {
  generateDraft: {
    url: `${agentApiBaseUrl}/v1/report-drafts/generate`,
    method: 'POST',
    requestType: 'GenerateReportDraftRequest',
    responseType: 'GenerateReportDraftResponse',
  },
  saveDraft: {
    url: `${apiBaseUrl}/v1/report-drafts`,
    method: 'POST',
    requestType: 'SaveReportDraftRequest',
    responseType: 'ReportDraftRecord',
  },
  fetchAnnotationTags: {
    url: `${agentApiBaseUrl}/v1/annotation-tags/infer`,
    method: 'POST',
    requestType: 'FetchAnnotationTagsRequest',
    responseType: 'AnnotationTag[]',
  },
}

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

const mapRawPatientRecord = (rawRecord: RawPatientRecord): PatientRecord => ({
  patientId: rawRecord.patient_id,
  patientName: rawRecord.patient_name,
  gender: rawRecord.gender,
  age: rawRecord.age,
  examDate: rawRecord.exam_date,
  status: rawRecord.status,
})

const mockRawPatients: RawPatientRecord[] = [
  {
    patient_id: 'EIS-2026-000128',
    patient_name: '张明远',
    gender: '男',
    age: 58,
    exam_date: '2026-04-16',
    status: 1,
  },
  {
    patient_id: 'EIS-2026-000129',
    patient_name: '李若兰',
    gender: '女',
    age: 46,
    exam_date: '2026-04-15',
    status: 2,
  },
  {
    patient_id: 'EIS-2026-000130',
    patient_name: '周航',
    gender: '男',
    age: 63,
    exam_date: '2026-04-16',
    status: 0,
  },
]

const primaryMockPatient: RawPatientRecord = mockRawPatients[0] ?? {
  patient_id: 'EIS-2026-000000',
  patient_name: '默认患者',
  gender: '其他',
  age: 0,
  exam_date: '2026-04-16',
  status: 0,
}

const mockVideoMaskData: PolygonMask[] = [
  {
    id: 'mask-frame-1',
    frameWidth: 1280,
    frameHeight: 720,
    fillColor: 'rgba(37, 99, 235, 0.26)',
    strokeColor: 'rgba(37, 99, 235, 0.9)',
    points: [
      [438, 212],
      [522, 188],
      [610, 218],
      [642, 306],
      [584, 368],
      [474, 346],
      [426, 272],
    ],
  },
]

const mockTumorMaskData: PolygonMask[] = [
  {
    id: 'tumor-roi-1',
    frameWidth: 1200,
    frameHeight: 900,
    fillColor: 'rgba(16, 185, 129, 0.28)',
    strokeColor: 'rgba(5, 150, 105, 0.95)',
    points: [
      [364, 266],
      [520, 216],
      [704, 274],
      [748, 436],
      [640, 584],
      [438, 612],
      [320, 480],
    ],
  },
]

const mockAnnotationTags: AnnotationTag[] = [
  {
    id: 'tag-1',
    label: '管状腺瘤',
    confidence: 0.92,
    targetTime: 12.5,
    locationLabel: '乙状结肠',
    needsReview: false,
  },
  {
    id: 'tag-2',
    label: 'Paris 0-Is',
    confidence: 0.86,
    targetTime: 12.8,
    locationLabel: '距肛缘 28 cm',
    needsReview: false,
  },
  {
    id: 'tag-3',
    label: '边界需复核',
    confidence: 0.67,
    targetTime: 13.1,
    locationLabel: '病灶远端侧',
    needsReview: true,
  },
]

const mockTumorDetails: TumorDetails = {
  estimatedSizeMm: 6.4,
  classification: '疑似管状腺瘤',
  location: '乙状结肠距肛缘约 28 cm',
  surfacePattern: '表面细颗粒样，边缘轻度隆起',
  confidence: 0.92,
}

const mockAgentDraftResponse: GenerateReportDraftResponse = {
  findings:
    '肠道准备尚可，镜下进至回盲部。乙状结肠距肛缘约 28 cm 处见一枚约 6 mm 广基隆起性病灶，表面细颗粒样，边界较清，NBI 下腺管结构轻度不规则。余结肠及直肠黏膜未见明确新发出血或活动性溃疡。',
  conclusion: '乙状结肠息肉，形态倾向腺瘤性病变，建议结合病理结果并安排常规随访。',
  layoutSuggestion: '正文建议按进镜范围、肠道准备、病灶部位与形态、处理建议四段排布，诊断结论单列。',
}

export const getPatientPreviewCards = async (): Promise<PatientRecord[]> => {
  await wait(180)
  return mockRawPatients.map(mapRawPatientRecord)
}

export const getReportBuilderMockContext = async (reportId?: string): Promise<ReportContextData> => {
  await wait(260)

  return {
    patient: mapRawPatientRecord(primaryMockPatient),
    videoSrc: '',
    maskData: mockVideoMaskData,
    showMask: true,
    videoFrameData: {
      frameId: reportId ? `${reportId}-frame-001` : 'frame-001',
      sourceId: 'scope-session-20260416-01',
      timestamp: 12.5,
      width: 1280,
      height: 720,
      suspectedLocation: '乙状结肠',
    },
    captureImageSrcs: ['/images/endoscopy-frame-demo.svg', '/images/tumor-roi-demo.svg'],
    reportSnippet: '乙状结肠见一枚约 6 mm 隆起性病灶，边界尚清，建议结合病理。',
    initialOpinion: '请基于抓拍图、视频分割结果和病灶部位，生成符合 EIS 规范的内镜报告草稿。',
    tumorFocus: {
      tumorImageSrc: '/images/tumor-roi-demo.svg',
      maskData: mockTumorMaskData,
      details: mockTumorDetails,
    },
  }
}

export const invokeReportDraftAgent = async (
  request: GenerateReportDraftRequest,
  onChunk?: (chunk: string) => void,
): Promise<GenerateReportDraftResponse> => {
  const streamChunks = [
    `正在分析患者 ${request.contextData.patient.patientName} 的检查上下文...\n`,
    '已提取病灶位置、形态及 NBI 特征。\n',
    '正在组织检查所见与诊断结论。\n',
    '已生成符合 EIS 排版习惯的结构化草稿。',
  ]

  for (const chunk of streamChunks) {
    await wait(320)
    onChunk?.(chunk)
  }

  await wait(240)
  return mockAgentDraftResponse
}

export const fetchSmartAnnotationTags = async (
  _request: FetchAnnotationTagsRequest,
): Promise<AnnotationTag[]> => {
  await wait(220)
  return mockAnnotationTags
}

export const saveReportDraft = async (
  request: SaveReportDraftRequest,
): Promise<ReportDraftRecord> => {
  await wait(260)

  return {
    ...request,
    reportId: request.reportId ?? 'draft-20260416-001',
    updatedAt: new Date().toISOString(),
  }
}