import axios from 'axios'

import type {
  ExemplarBankDecision,
  ExemplarBankRequest,
  ExemplarFeedbackRequest,
  ExemplarFeedbackResult,
  ExemplarRetrievalRequest,
  ExemplarRetrievalResult,
  ExpertConfiguration,
  WorkspacePatient,
  WorkspaceReportRequest,
  WorkspaceReportResult,
  WorkspaceSegmentation,
} from '@/types/workspace'

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
  retrieval_applied?: boolean
  retrieval_confidence?: number | null
  retrieval_uncertainty?: number | null
  retrieval_candidate_count?: number
  retrieval_bank_id?: string | null
  retrieval_prior_keys?: string[]
}

/**
 * brief:
 *   Handle workspace client.
 *
 * parameter:
 *   - response: Input value for response.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const workspaceClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 45000,
})

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
 *   Handle calculate polygon area.
 *
 * parameter:
 *   - points: Input value for points.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const calculatePolygonArea = (points: [number, number][]) => {
  if (points.length < 3) {
    return 0
  }

  let area = 0
  for (let index = 0; index < points.length; index += 1) {
    /**
     * brief:
     *   Handle current point.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const currentPoint = points[index]
    /**
     * brief:
     *   Handle next point.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const nextPoint = points[(index + 1) % points.length]
    if (!currentPoint || !nextPoint) {
      continue
    }

    const [x1, y1] = currentPoint
    const [x2, y2] = nextPoint
    area += x1 * y2 - x2 * y1
  }

  return Math.abs(area) / 2
}

/**
 * brief:
 *   Handle segment workspace image.
 *
 * parameter:
 *   - file: Input value for file.
 *   - dimensions: Input value for dimensions.
 *   - context: Input value for context.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const segmentWorkspaceImage = async (
  file: File,
  dimensions: { width: number; height: number },
  context?: {
    patient: WorkspacePatient
    expertConfig: ExpertConfiguration
    bankId?: string
    topK?: number
  },
): Promise<WorkspaceSegmentation> => {
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
  formData.append('image', file, file.name)
  if (context) {
    formData.append('patient_payload', JSON.stringify(context.patient))
    formData.append('expert_config_payload', JSON.stringify(context.expertConfig))
    formData.append('bank_id', context.bankId ?? 'default-bank')
    formData.append('retrieval_top_k', String(context.topK ?? 6))
  }

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
  const response = await workspaceClient.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    '/analysis/segment-frame',
    formData,
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
  /**
   * brief:
   *   Handle mask area pixels.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const maskAreaPixels = payload.mask_area_pixels ?? calculatePolygonArea(payload.mask_coordinates)
  /**
   * brief:
   *   Handle image area.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const imageArea = Math.max(dimensions.width * dimensions.height, 1)

  return {
    maskDataUrl: payload.mask_data_url,
    maskCoordinates: payload.mask_coordinates,
    boundingBox: payload.bounding_box,
    maskAreaPixels,
    maskAreaRatio: maskAreaPixels / imageArea,
    pointCount: payload.mask_coordinates.length,
    retrievalApplied: payload.retrieval_applied ?? false,
    retrievalConfidence: payload.retrieval_confidence ?? null,
    retrievalUncertainty: payload.retrieval_uncertainty ?? null,
    retrievalCandidateCount: payload.retrieval_candidate_count ?? 0,
    retrievalBankId: payload.retrieval_bank_id ?? null,
    retrievalPriorKeys: payload.retrieval_prior_keys ?? [],
  }
}

/**
 * brief:
 *   Generate workspace report.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const generateWorkspaceReport = async (
  payload: WorkspaceReportRequest,
): Promise<WorkspaceReportResult> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - payload: Input value for payload.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await workspaceClient.post<ApiResponseEnvelope<WorkspaceReportResult>>(
    '/agent/workspace/report',
    payload,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Evaluate exemplar candidate.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const evaluateExemplarCandidate = async (
  payload: ExemplarBankRequest,
): Promise<ExemplarBankDecision> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - payload: Input value for payload.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarBankDecision>>(
    '/agent/workspace/exemplar-bank',
    payload,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Retrieve exemplar prior.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const retrieveExemplarPrior = async (
  payload: ExemplarRetrievalRequest,
): Promise<ExemplarRetrievalResult> => {
  /**
   * brief:
   *   Handle response.
   *
   * parameter:
   *   - payload: Input value for payload.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarRetrievalResult>>(
    '/agent/workspace/exemplar-bank/retrieve-prior',
    payload,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Send exemplar feedback.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const sendExemplarFeedback = async (
  payload: ExemplarFeedbackRequest,
): Promise<ExemplarFeedbackResult> => {
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
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarFeedbackResult>>(
    '/agent/workspace/exemplar-bank/feedback',
    payload,
  )

  return extractApiData(response)
}

/**
 * brief:
 *   Preload workspace sam3 model.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const preloadWorkspaceSam3Model = async (): Promise<void> => {
  await workspaceClient.post('/analysis/preload-model')
}
