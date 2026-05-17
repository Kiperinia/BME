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

const workspaceClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 45000,
})

const extractApiData = <T>(response: { data: ApiResponseEnvelope<T> }) => response.data.data

const calculatePolygonArea = (points: [number, number][]) => {
  if (points.length < 3) {
    return 0
  }

  let area = 0
  for (let index = 0; index < points.length; index += 1) {
    const currentPoint = points[index]
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
  const formData = new FormData()
  formData.append('image', file, file.name)
  if (context) {
    formData.append('patient_payload', JSON.stringify(context.patient))
    formData.append('expert_config_payload', JSON.stringify(context.expertConfig))
    formData.append('bank_id', context.bankId ?? 'default-bank')
    formData.append('retrieval_top_k', String(context.topK ?? 6))
  }

  const response = await workspaceClient.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    '/analysis/segment-frame',
    formData,
  )

  const payload = extractApiData(response)
  const maskAreaPixels = payload.mask_area_pixels ?? calculatePolygonArea(payload.mask_coordinates)
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

export const generateWorkspaceReport = async (
  payload: WorkspaceReportRequest,
): Promise<WorkspaceReportResult> => {
  const response = await workspaceClient.post<ApiResponseEnvelope<WorkspaceReportResult>>(
    '/agent/workspace/report',
    payload,
  )

  return extractApiData(response)
}

export const evaluateExemplarCandidate = async (
  payload: ExemplarBankRequest,
): Promise<ExemplarBankDecision> => {
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarBankDecision>>(
    '/agent/workspace/exemplar-bank',
    payload,
  )

  return extractApiData(response)
}

export const retrieveExemplarPrior = async (
  payload: ExemplarRetrievalRequest,
): Promise<ExemplarRetrievalResult> => {
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarRetrievalResult>>(
    '/agent/workspace/exemplar-bank/retrieve-prior',
    payload,
  )

  return extractApiData(response)
}

export const sendExemplarFeedback = async (
  payload: ExemplarFeedbackRequest,
): Promise<ExemplarFeedbackResult> => {
  const response = await workspaceClient.post<ApiResponseEnvelope<ExemplarFeedbackResult>>(
    '/agent/workspace/exemplar-bank/feedback',
    payload,
  )

  return extractApiData(response)
}

export const preloadWorkspaceSam3Model = async (): Promise<void> => {
  await workspaceClient.post('/analysis/preload-model')
}
