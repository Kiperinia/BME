import axios from 'axios'

import type { SystemSettingsPayload, SystemSettingsResponse } from '@/types/systemSettings'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'

interface ApiResponseEnvelope<T> {
  code: number
  message: string
  data: T
}

const systemSettingsClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 30000,
})

export const getSystemSettings = async (): Promise<SystemSettingsResponse> => {
  const response = await systemSettingsClient.get<ApiResponseEnvelope<SystemSettingsResponse>>('/system/settings')
  return response.data.data
}

export const updateSystemSettings = async (
  payload: SystemSettingsPayload,
): Promise<SystemSettingsResponse> => {
  const response = await systemSettingsClient.put<ApiResponseEnvelope<SystemSettingsResponse>>(
    '/system/settings',
    payload,
  )
  return response.data.data
}