import axios from 'axios'

import type {
  ExemplarBankDecision,
  ExemplarBankRequest,
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
  mask_coordinates: [number, number][]
  bounding_box: [number, number, number, number]
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
): Promise<WorkspaceSegmentation> => {
  const formData = new FormData()
  formData.append('image', file, file.name)

  const response = await workspaceClient.post<ApiResponseEnvelope<SegmentFrameApiPayload>>(
    '/analysis/segment-frame',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    },
  )

  const payload = extractApiData(response)
  const maskAreaPixels = calculatePolygonArea(payload.mask_coordinates)
  const imageArea = Math.max(dimensions.width * dimensions.height, 1)

  return {
    maskCoordinates: payload.mask_coordinates,
    boundingBox: payload.bounding_box,
    maskAreaPixels,
    maskAreaRatio: maskAreaPixels / imageArea,
    pointCount: payload.mask_coordinates.length,
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
