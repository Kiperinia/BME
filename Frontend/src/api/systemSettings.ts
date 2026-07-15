import axios from 'axios'

import type { SystemSettingsPayload, SystemSettingsResponse } from '@/types/systemSettings'

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

/**
 * brief:
 *   Handle system settings client.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const systemSettingsClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 30000,
})

/**
 * brief:
 *   Handle get system settings.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const getSystemSettings = async (): Promise<SystemSettingsResponse> => {
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
  const response = await systemSettingsClient.get<ApiResponseEnvelope<SystemSettingsResponse>>('/system/settings')
  return response.data.data
}

/**
 * brief:
 *   Update system settings.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const updateSystemSettings = async (
  payload: SystemSettingsPayload,
): Promise<SystemSettingsResponse> => {
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
  const response = await systemSettingsClient.put<ApiResponseEnvelope<SystemSettingsResponse>>(
    '/system/settings',
    payload,
  )
  return response.data.data
}
