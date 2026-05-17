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
  ReportContextData,
  ReportDraftRecord,
  SaveReportDraftRequest,
  SegmentFrameResponse,
} from '@/types/eis'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'
const agentApiBaseUrl = import.meta.env.VITE_AGENT_API_BASE_URL ?? `${apiBaseUrl}/agent`

export const httpClient = axios.create({
  timeout: 45000,
})

const SAM3_READY_POLL_INTERVAL_MS = 1000
const SAM3_READY_MAX_WAIT_MS = 30000
const SAM3_SEGMENT_TIMEOUT_MS = 120000

interface ApiResponseEnvelope<T> {
  code: number
  message: string
  data: T
}

interface SegmentFrameApiPayload {
  mask_data_url: string
  mask_coordinates: [number, number][]
  bounding_box: [number, number, number, number]
  mask_area_pixels: number
}

interface Sam3PreloadStatus {
  started: boolean
  ready: boolean
  in_progress: boolean
  load_mode: string
  device: string
  warmup_enabled: boolean
  last_error: string
}

export const reportBuilderApiContracts = {
  fetchPatientPreviews: {
    url: `${agentApiBaseUrl}/patient-previews`,
    method: 'GET',
    requestType: 'void',
    responseType: 'PatientRecord[]',
  },
  fetchReportContext: {
    url: `${agentApiBaseUrl}/report-context`,
    method: 'GET',
    requestType: '{ reportId?: string; patientId?: string }',
    responseType: 'ReportContextData',
  },
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
} satisfies Record<
  'fetchPatientPreviews' | 'fetchReportContext' | 'generateDraft' | 'saveDraft' | 'fetchAnnotationTags' | 'segmentFrame',
  ApiContractDefinition
>

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
  const width = image.naturalWidth || 1024
  const height = image.naturalHeight || 1024
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

export const getPatientPreviewCards = async (): Promise<PatientRecord[]> => {
  const response = await httpClient.get<ApiResponseEnvelope<PatientRecord[]>>(
    reportBuilderApiContracts.fetchPatientPreviews.url,
  )

  return extractApiData(response)
}

export const getReportBuilderContext = async (
  reportId?: string,
  patientId?: string,
): Promise<ReportContextData> => {
  const response = await httpClient.get<ApiResponseEnvelope<ReportContextData>>(
    reportBuilderApiContracts.fetchReportContext.url,
    {
      params: {
        reportId,
        patientId,
      },
    },
  )

  return extractApiData(response)
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
  await ensureSam3Ready()
  const normalizedImageSource = await normalizeImageSource(imageSource)
  const imageBlob = dataUrlToBlob(normalizedImageSource)
  const formData = new FormData()
  formData.append('image', imageBlob, 'captured-frame.png')

  const response = await httpClient.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    reportBuilderApiContracts.segmentFrame.url,
    formData,
    { timeout: SAM3_SEGMENT_TIMEOUT_MS },
  )

  const payload = extractApiData(response)
  return {
    maskDataUrl: payload.mask_data_url,
    maskCoordinates: payload.mask_coordinates,
    boundingBox: payload.bounding_box,
    maskAreaPixels: payload.mask_area_pixels,
  }
}

export const preloadSam3Model = async (): Promise<void> => {
  await httpClient.post(`${apiBaseUrl}/analysis/preload-model`)
}

const getSam3PreloadStatus = async (): Promise<Sam3PreloadStatus> => {
  const response = await httpClient.get<ApiResponseEnvelope<Sam3PreloadStatus>>(
    `${apiBaseUrl}/analysis/preload-model-status`,
  )
  return extractApiData(response)
}

const ensureSam3Ready = async (): Promise<void> => {
  await preloadSam3Model()
  const deadline = Date.now() + SAM3_READY_MAX_WAIT_MS

  while (Date.now() < deadline) {
    const status = await getSam3PreloadStatus()
    if (status.ready) {
      return
    }

    if (status.last_error) {
      throw new Error(`SAM3 preload failed: ${status.last_error}`)
    }

    await wait(SAM3_READY_POLL_INTERVAL_MS)
  }
}