export type LlmProviderKind = 'openai_compatible' | 'modelscope'
export type Sam3RuntimeMode = 'mock' | 'sam3'
export type Sam3LoraStage = 'stage_a' | 'stage_b' | 'stage_c'

export interface LlmProfile {
  profileId: string
  providerKind: LlmProviderKind
  defaultProvider: string
  defaultModel: string
  apiKey: string
  baseUrl: string
  timeout: number
}

export interface LlmSettings {
  activeProfile: string
  profiles: LlmProfile[]
}

export interface AgentSettings {
  enableLlm: boolean
  enableLlmReport: boolean
  pixelSizeMm: number
}

export interface Sam3Settings {
  loadMode: Sam3RuntimeMode
  device: string
  checkpointPath: string
  inputSize: number
  keepAspectRatio: boolean
  warmupEnabled: boolean
  loraEnabled: boolean
  loraPath: string
  loraStage: Sam3LoraStage
}

export interface RuntimeSettings {
  inferenceTimeoutSeconds: number
  maxUploadSizeMb: number
  mockDelayMs: number
}

export interface SystemSettingsPayload {
  llm: LlmSettings
  agent: AgentSettings
  sam3: Sam3Settings
  runtime: RuntimeSettings
}

export interface SystemSettingsStatus {
  llmReady: boolean
  sam3Ready: boolean
  sam3RuntimeMode: Sam3RuntimeMode
  loraLoaded: boolean
  llmConfigPath: string
  runtimeSettingsPath: string
  warnings: string[]
}

export interface SystemSettingsResponse {
  settings: SystemSettingsPayload
  status: SystemSettingsStatus
}
