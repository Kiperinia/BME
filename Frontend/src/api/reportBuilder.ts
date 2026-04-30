import axios from 'axios'

import type {
  AgentWorkflowSummary,
  AnnotationTag,
  ApiContractDefinition,
  FetchAnnotationTagsResponse,
  FetchAnnotationTagsRequest,
  GenerateReportDraftRequest,
  GenerateReportDraftResponse,
  PatientRecord,
  PolygonMask,
  RawPatientRecord,
  ReportContextData,
  ReportDraftRecord,
  SaveReportDraftRequest,
  SegmentFrameResponse,
  TumorDetails,
} from '@/types/eis'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'
const agentApiBaseUrl = import.meta.env.VITE_AGENT_API_BASE_URL ?? `${apiBaseUrl}/agent`

export const httpClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 45000,
})

interface ApiResponseEnvelope<T> {
  code: number
  message: string
  data: T
}

interface SegmentFrameApiPayload {
  mask_coordinates: [number, number][]
  bounding_box: [number, number, number, number]
}

export const reportBuilderApiContracts = {
  generateDraft: {
    url: `${agentApiBaseUrl}/report-drafts/generate`,
    method: 'POST',
    requestType: 'GenerateReportDraftRequest',
    responseType: 'GenerateReportDraftResponse',
  },
  saveDraft: {
    url: `${agentApiBaseUrl}/report-drafts`,
    method: 'POST',
    requestType: 'SaveReportDraftRequest',
    responseType: 'ReportDraftRecord',
  },
  fetchAnnotationTags: {
    url: `${agentApiBaseUrl}/annotation-tags/infer`,
    method: 'POST',
    requestType: 'FetchAnnotationTagsRequest',
    responseType: 'FetchAnnotationTagsResponse',
  },
  segmentFrame: {
    url: `${apiBaseUrl}/analysis/segment-frame`,
    method: 'POST',
    requestType: 'multipart/form-data',
    responseType: 'SegmentFrameResponse',
  },
} satisfies Record<'generateDraft' | 'saveDraft' | 'fetchAnnotationTags' | 'segmentFrame', ApiContractDefinition>

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

const extractApiData = <T>(response: { data: ApiResponseEnvelope<T> }) => response.data.data

const loadImage = (src: string) => new Promise<HTMLImageElement>((resolve, reject) => {
  const image = new Image()
  image.crossOrigin = 'anonymous'
  image.onload = () => resolve(image)
  image.onerror = () => reject(new Error(`failed to load image source: ${src}`))
  image.src = src
})

const rasterizeImageSource = async (source: string): Promise<string> => {
  if (!source) {
    return source
  }

  const isRasterDataUrl = source.startsWith('data:image/') && !source.startsWith('data:image/svg+xml')
  const needsCanvasRasterization = source.endsWith('.svg') || source.startsWith('data:image/svg+xml')

  if (isRasterDataUrl && !needsCanvasRasterization) {
    return source
  }

  const image = await loadImage(source)
  const width = image.naturalWidth || 1280
  const height = image.naturalHeight || 720
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('failed to create canvas context for rasterization')
  }

  canvas.width = width
  canvas.height = height
  context.drawImage(image, 0, 0, width, height)

  return canvas.toDataURL('image/png')
}

const normalizeImageSource = async (source: string) => {
  if (!source) {
    return source
  }

  if (source.startsWith('data:image/') && !source.startsWith('data:image/svg+xml')) {
    return source
  }

  return rasterizeImageSource(source)
}

const dataUrlToBlob = (dataUrl: string) => {
  const [header, encodedPayload] = dataUrl.split(',', 2)
  if (!header || !encodedPayload) {
    throw new Error('invalid image data url')
  }
  const mimeType = header.match(/data:(.*?);base64/i)?.[1] ?? 'image/png'
  const binary = window.atob(encodedPayload)
  const buffer = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    buffer[index] = binary.charCodeAt(index)
  }

  return new Blob([buffer], { type: mimeType })
}

const prepareReportContextForAgent = async (contextData: ReportContextData): Promise<ReportContextData> => {
  const [tumorImageSrc, captureImageSrcs] = await Promise.all([
    normalizeImageSource(contextData.tumorFocus.tumorImageSrc),
    Promise.all(contextData.captureImageSrcs.map((imageSrc) => normalizeImageSource(imageSrc))),
  ])

  return {
    ...contextData,
    captureImageSrcs,
    tumorFocus: {
      ...contextData.tumorFocus,
      tumorImageSrc,
    },
  }
}

const streamAgentMessages = async (
  workflow: AgentWorkflowSummary,
  onChunk?: (chunk: string) => void,
) => {
  for (const message of workflow.steps) {
    onChunk?.(`${message}\n`)
    await wait(120)
  }

  for (const warning of workflow.warnings) {
    onChunk?.(`注意：${warning}\n`)
    await wait(120)
  }
}

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

const mockTumorDetails: TumorDetails = {
  estimatedSizeMm: 6.4,
  classification: '疑似管状腺瘤',
  location: '乙状结肠距肛缘约 28 cm',
  surfacePattern: '表面细颗粒样，边缘轻度隆起',
  confidence: 0.92,
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
  const preparedContext = await prepareReportContextForAgent(request.contextData)
  const response = await httpClient.post<ApiResponseEnvelope<GenerateReportDraftResponse>>(
    reportBuilderApiContracts.generateDraft.url,
    {
      ...request,
      contextData: preparedContext,
    },
  )
  const payload = extractApiData(response)

  await streamAgentMessages(payload.workflow, onChunk)
  return payload
}

export const fetchSmartAnnotationTags = async (
  request: FetchAnnotationTagsRequest,
): Promise<FetchAnnotationTagsResponse> => {
  const preparedContext = await prepareReportContextForAgent(request.contextData)
  const response = await httpClient.post<ApiResponseEnvelope<FetchAnnotationTagsResponse>>(
    reportBuilderApiContracts.fetchAnnotationTags.url,
    {
      ...request,
      contextData: preparedContext,
    },
  )

  return extractApiData(response)
}

export const saveReportDraft = async (
  request: SaveReportDraftRequest,
): Promise<ReportDraftRecord> => {
  const response = await httpClient.post<ApiResponseEnvelope<ReportDraftRecord>>(
    reportBuilderApiContracts.saveDraft.url,
    request,
  )

  return extractApiData(response)
}

export const segmentFrameWithSam3 = async (imageSource: string): Promise<SegmentFrameResponse> => {
  const normalizedImageSource = await normalizeImageSource(imageSource)
  const imageBlob = dataUrlToBlob(normalizedImageSource)
  const formData = new FormData()
  formData.append('image', imageBlob, 'captured-frame.png')

  const response = await axios.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    reportBuilderApiContracts.segmentFrame.url,
    formData,
    {
      timeout: 45000,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    },
  )

  const payload = extractApiData(response)
  return {
    maskCoordinates: payload.mask_coordinates,
    boundingBox: payload.bounding_box,
  }
}