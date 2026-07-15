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

/**
 * brief:
 *   Handle api base url.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'
/**
 * brief:
 *   Handle agent api base url.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const agentApiBaseUrl = import.meta.env.VITE_AGENT_API_BASE_URL ?? `${apiBaseUrl}/agent`

/**
 * brief:
 *   Handle http client.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const httpClient = axios.create({
  timeout: 45000,
})

/**
 * brief:
 *   Handle sam3 ready poll interval ms.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const SAM3_READY_POLL_INTERVAL_MS = 1000
/**
 * brief:
 *   Handle sam3 ready max wait ms.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const SAM3_READY_MAX_WAIT_MS = 30000
/**
 * brief:
 *   Handle sam3 segment timeout ms.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle report builder api contracts.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle wait.
 *
 * parameter:
 *   - ms: Input value for ms.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

/**
 * brief:
 *   Handle extract api data.
 *
 * parameter:
 *   - response: Input value for response.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const extractApiData = <T>(response: { data: ApiResponseEnvelope<T> }) => response.data.data

/**
 * brief:
 *   Load image.
 *
 * parameter:
 *   - src: Input value for src.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const loadImage = (src: string) => new Promise<HTMLImageElement>((resolve, reject) => {
  /**
   * brief:
   *   Handle image.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const image = new Image()
  image.crossOrigin = 'anonymous'
  image.onload = () => resolve(image)
  image.onerror = () => reject(new Error(`failed to load image source: ${src}`))
  image.src = src
})

/**
 * brief:
 *   Handle rasterize image source.
 *
 * parameter:
 *   - source: Input value for source.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const rasterizeImageSource = async (source: string): Promise<string> => {
  if (!source) {
    return source
  }

  /**
   * brief:
   *   Handle is raster data url.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isRasterDataUrl = source.startsWith('data:image/') && !source.startsWith('data:image/svg+xml')
  /**
   * brief:
   *   Handle needs canvas rasterization.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const needsCanvasRasterization = source.endsWith('.svg') || source.startsWith('data:image/svg+xml')

  if (isRasterDataUrl && !needsCanvasRasterization) {
    return source
  }

  /**
   * brief:
   *   Handle image.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const image = await loadImage(source)
  /**
   * brief:
   *   Handle width.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const width = image.naturalWidth || 1024
  /**
   * brief:
   *   Handle height.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const height = image.naturalHeight || 1024
  /**
   * brief:
   *   Handle canvas.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const canvas = document.createElement('canvas')
  /**
   * brief:
   *   Handle context.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('failed to create canvas context for rasterization')
  }

  canvas.width = width
  canvas.height = height
  context.drawImage(image, 0, 0, width, height)

  return canvas.toDataURL('image/png')
}

/**
 * brief:
 *   Normalize image source.
 *
 * parameter:
 *   - source: Input value for source.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const normalizeImageSource = async (source: string) => {
  if (!source) {
    return source
  }

  if (source.startsWith('data:image/') && !source.startsWith('data:image/svg+xml')) {
    return source
  }

  return rasterizeImageSource(source)
}

/**
 * brief:
 *   Handle data url to blob.
 *
 * parameter:
 *   - dataUrl: Input value for dataUrl.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const dataUrlToBlob = (dataUrl: string) => {
  const [header, encodedPayload] = dataUrl.split(',', 2)
  if (!header || !encodedPayload) {
    throw new Error('invalid image data url')
  }
  /**
   * brief:
   *   Handle mime type.
   *
   * parameter:
   *   - contextData: Input value for contextData.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const mimeType = header.match(/data:(.*?);base64/i)?.[1] ?? 'image/png'
  /**
   * brief:
   *   Handle binary.
   *
   * parameter:
   *   - contextData: Input value for contextData.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const binary = window.atob(encodedPayload)
  /**
   * brief:
   *   Handle buffer.
   *
   * parameter:
   *   - contextData: Input value for contextData.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const buffer = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    buffer[index] = binary.charCodeAt(index)
  }

  return new Blob([buffer], { type: mimeType })
}

/**
 * brief:
 *   Prepare report context for agent.
 *
 * parameter:
 *   - contextData: Input value for contextData.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle stream agent messages.
 *
 * parameter:
 *   - workflow: Input value for workflow.
 *   - onChunk: Input value for onChunk.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle get patient preview cards.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const getPatientPreviewCards = async (): Promise<PatientRecord[]> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - reportId: Input value for reportId.
   *   - patientId: Input value for patientId.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.get<ApiResponseEnvelope<PatientRecord[]>>(
    reportBuilderApiContracts.fetchPatientPreviews.url,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Handle get report builder context.
 *
 * parameter:
 *   - reportId: Input value for reportId.
 *   - patientId: Input value for patientId.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const getReportBuilderContext = async (
  reportId?: string,
  patientId?: string,
): Promise<ReportContextData> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
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

/**
 * brief:
 *   Handle invoke report draft agent.
 *
 * parameter:
 *   - request: Input value for request.
 *   - onChunk: Input value for onChunk.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const invokeReportDraftAgent = async (
  request: GenerateReportDraftRequest,
  onChunk?: (chunk: string) => void,
): Promise<GenerateReportDraftResponse> => {
  /**
   * brief:
   *   Handle prepared context.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const preparedContext = await prepareReportContextForAgent(request.contextData)
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.post<ApiResponseEnvelope<GenerateReportDraftResponse>>(
    reportBuilderApiContracts.generateDraft.url,
    {
      ...request,
      contextData: preparedContext,
    },
  )
  /**
   * brief:
   *   Handle payload.
   *
   * parameter:
   *   - request: Input value for request.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const payload = extractApiData(response)

  await streamAgentMessages(payload.workflow, onChunk)
  return payload
}

/**
 * brief:
 *   Fetch smart annotation tags.
 *
 * parameter:
 *   - request: Input value for request.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const fetchSmartAnnotationTags = async (
  request: FetchAnnotationTagsRequest,
): Promise<FetchAnnotationTagsResponse> => {
  /**
   * brief:
   *   Handle prepared context.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const preparedContext = await prepareReportContextForAgent(request.contextData)
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.post<ApiResponseEnvelope<FetchAnnotationTagsResponse>>(
    reportBuilderApiContracts.fetchAnnotationTags.url,
    {
      ...request,
      contextData: preparedContext,
    },
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Save report draft.
 *
 * parameter:
 *   - request: Input value for request.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const saveReportDraft = async (
  request: SaveReportDraftRequest,
): Promise<ReportDraftRecord> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - imageSource: Input value for imageSource.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.post<ApiResponseEnvelope<ReportDraftRecord>>(
    reportBuilderApiContracts.saveDraft.url,
    request,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Handle segment frame with sam3.
 *
 * parameter:
 *   - imageSource: Input value for imageSource.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const segmentFrameWithSam3 = async (imageSource: string): Promise<SegmentFrameResponse> => {
  await ensureSam3Ready()
  /**
   * brief:
   *   Handle normalized image source.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const normalizedImageSource = await normalizeImageSource(imageSource)
  /**
   * brief:
   *   Handle image blob.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const imageBlob = dataUrlToBlob(normalizedImageSource)
  /**
   * brief:
   *   Handle form data.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const formData = new FormData()
  formData.append('image', imageBlob, 'captured-frame.png')

  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    reportBuilderApiContracts.segmentFrame.url,
    formData,
    { timeout: SAM3_SEGMENT_TIMEOUT_MS },
  )

  /**
   * brief:
   *   Handle payload.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const payload = extractApiData(response)
  return {
    maskDataUrl: payload.mask_data_url,
    maskCoordinates: payload.mask_coordinates,
    boundingBox: payload.bounding_box,
    maskAreaPixels: payload.mask_area_pixels,
  }
}

/**
 * brief:
 *   Preload sam3 model.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const preloadSam3Model = async (): Promise<void> => {
  await httpClient.post(`${apiBaseUrl}/analysis/preload-model`)
}

/**
 * brief:
 *   Handle get sam3 preload status.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const getSam3PreloadStatus = async (): Promise<Sam3PreloadStatus> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await httpClient.get<ApiResponseEnvelope<Sam3PreloadStatus>>(
    `${apiBaseUrl}/analysis/preload-model-status`,
  )
  return extractApiData(response)
}

/**
 * brief:
 *   Handle ensure sam3 ready.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const ensureSam3Ready = async (): Promise<void> => {
  await preloadSam3Model()
  /**
   * brief:
   *   Handle deadline.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const deadline = Date.now() + SAM3_READY_MAX_WAIT_MS

  while (Date.now() < deadline) {
    /**
     * brief:
     *   Handle status.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
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
