<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { getSystemSettings, updateSystemSettings } from '@/api/systemSettings'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import type {
  LlmProfile,
  LlmProviderKind,
  SystemSettingsPayload,
  SystemSettingsStatus,
} from '@/types/systemSettings'

const savedSettings = ref<SystemSettingsPayload | null>(null)
const form = ref<SystemSettingsPayload | null>(null)
const runtimeStatus = ref<SystemSettingsStatus | null>(null)
const isLoading = ref(false)
const isSaving = ref(false)
const loadErrorMessage = ref('')
const toastVisible = ref(false)
const toastMessage = ref('')
const toastTone = ref<'info' | 'success' | 'error'>('info')

let toastTimer: number | undefined

const textInputClass = 'mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-sky-400 dark:border-slate-700 dark:bg-slate-950 dark:text-white'
const selectInputClass = `${textInputClass} appearance-none pr-10`

const cloneSettings = (value: SystemSettingsPayload): SystemSettingsPayload => {
  return JSON.parse(JSON.stringify(value)) as SystemSettingsPayload
}

const buildUniqueProfileId = (baseId: string, existingIds: string[]) => {
  let nextId = baseId
  let suffix = 2

  while (existingIds.includes(nextId)) {
    nextId = `${baseId}-${suffix}`
    suffix += 1
  }

  return nextId
}

const createProfileDraft = (providerKind: LlmProviderKind, profileId: string): LlmProfile => {
  if (providerKind === 'modelscope') {
    return {
      profileId,
      providerKind,
      defaultProvider: 'modelscope',
      defaultModel: 'Qwen/Qwen2.5-VL-72B-Instruct',
      apiKey: '',
      baseUrl: 'https://api-inference.modelscope.cn/v1/',
      timeout: 60,
    }
  }

  return {
    profileId,
    providerKind,
    defaultProvider: 'openai',
    defaultModel: 'gpt-4o-mini',
    apiKey: '',
    baseUrl: '',
    timeout: 60,
  }
}

const showToast = (message: string, tone: 'info' | 'success' | 'error' = 'info') => {
  toastMessage.value = message
  toastTone.value = tone
  toastVisible.value = true

  if (toastTimer) {
    window.clearTimeout(toastTimer)
  }

  toastTimer = window.setTimeout(() => {
    toastVisible.value = false
  }, 2800)
}

const isDirty = computed(() => {
  if (!savedSettings.value || !form.value) {
    return false
  }

  return JSON.stringify(savedSettings.value) !== JSON.stringify(form.value)
})

const activeLlmProfile = computed(() => {
  if (!form.value) {
    return null
  }

  return form.value.llm.profiles.find((profile) => profile.profileId === form.value?.llm.activeProfile) ?? null
})

const activeProfileDescription = computed(() => {
  if (!activeLlmProfile.value) {
    return ''
  }

  return activeLlmProfile.value.providerKind === 'openai_compatible'
    ? '当前 Agent 将优先使用 OpenAI 兼容接口配置。'
    : '当前 Agent 将优先使用 ModelScope 配置。'
})

const activeProfileKindLabel = computed(() => {
  if (!activeLlmProfile.value) {
    return ''
  }

  return activeLlmProfile.value.providerKind === 'openai_compatible'
    ? 'OpenAI Compatible'
    : 'ModelScope'
})

const runtimeHeadline = computed(() => {
  if (!runtimeStatus.value) {
    return '未加载'
  }

  if (runtimeStatus.value.sam3RuntimeMode === 'mock') {
    return 'SAM3 Mock 联调模式'
  }

  return runtimeStatus.value.sam3Ready ? 'SAM3 实时推理已就绪' : 'SAM3 配置已保存，运行时未就绪'
})

const loadSettings = async () => {
  isLoading.value = true
  loadErrorMessage.value = ''

  try {
    const response = await getSystemSettings()
    savedSettings.value = cloneSettings(response.settings)
    form.value = cloneSettings(response.settings)
    runtimeStatus.value = response.status
  } catch (error) {
    const message = error instanceof Error ? error.message : '系统设置加载失败。'
    loadErrorMessage.value = message
  } finally {
    isLoading.value = false
  }
}

const handleReset = () => {
  if (!savedSettings.value) {
    return
  }

  form.value = cloneSettings(savedSettings.value)
  showToast('已恢复为当前已保存配置。')
}

const handleCreateProfile = (providerKind: LlmProviderKind) => {
  if (!form.value) {
    return
  }

  const baseId = providerKind === 'modelscope' ? 'modelscope-profile' : 'openai-profile'
  const profileId = buildUniqueProfileId(
    baseId,
    form.value.llm.profiles.map((profile) => profile.profileId),
  )

  form.value.llm.profiles.push(createProfileDraft(providerKind, profileId))
  form.value.llm.activeProfile = profileId
  showToast(`已创建新的 ${providerKind === 'modelscope' ? 'ModelScope' : 'OpenAI'} Profile。`, 'success')
}

const handleSetActiveProfile = (profileId: string) => {
  if (!form.value) {
    return
  }

  form.value.llm.activeProfile = profileId
}

const handleDeleteProfile = (profileId: string) => {
  if (!form.value) {
    return
  }

  if (form.value.llm.profiles.length === 1) {
    showToast('至少需要保留一个 Profile。', 'error')
    return
  }

  const nextProfiles = form.value.llm.profiles.filter((profile) => profile.profileId !== profileId)
  form.value.llm.profiles = nextProfiles

  if (form.value.llm.activeProfile === profileId) {
    form.value.llm.activeProfile = nextProfiles[0]?.profileId ?? ''
  }

  showToast(`已删除 Profile ${profileId}。`, 'info')
}

const handleRenameActiveProfile = (value: string) => {
  if (!form.value || !activeLlmProfile.value) {
    return
  }

  const nextId = value.trim().replace(/\s+/g, '_')
  const currentProfile = activeLlmProfile.value
  if (!nextId || nextId === currentProfile.profileId) {
    return
  }

  const hasConflict = form.value.llm.profiles.some((profile) => {
    return profile.profileId === nextId && profile !== currentProfile
  })

  if (hasConflict) {
    showToast(`Profile 标识 ${nextId} 已存在。`, 'error')
    return
  }

  const previousId = currentProfile.profileId
  currentProfile.profileId = nextId
  if (form.value.llm.activeProfile === previousId) {
    form.value.llm.activeProfile = nextId
  }
}

const handleProviderKindChange = (value: string) => {
  if (!activeLlmProfile.value) {
    return
  }

  const providerKind = value === 'modelscope' ? 'modelscope' : 'openai_compatible'
  activeLlmProfile.value.providerKind = providerKind

  if (providerKind === 'modelscope') {
    if (!activeLlmProfile.value.defaultProvider.trim()) {
      activeLlmProfile.value.defaultProvider = 'modelscope'
    }
    if (!activeLlmProfile.value.baseUrl.trim()) {
      activeLlmProfile.value.baseUrl = 'https://api-inference.modelscope.cn/v1/'
    }
    return
  }

  if (!activeLlmProfile.value.defaultProvider.trim()) {
    activeLlmProfile.value.defaultProvider = 'openai'
  }
}

const handleSave = async () => {
  if (!form.value) {
    return
  }

  isSaving.value = true

  try {
    const response = await updateSystemSettings(form.value)
    savedSettings.value = cloneSettings(response.settings)
    form.value = cloneSettings(response.settings)
    runtimeStatus.value = response.status
    showToast('系统设置已保存，并已尝试重载运行时。', 'success')
  } catch (error) {
    const message = error instanceof Error ? error.message : '系统设置保存失败。'
    showToast(message, 'error')
  } finally {
    isSaving.value = false
  }
}

onMounted(async () => {
  await loadSettings()
})
</script>

<template>
  <main class="mx-auto w-full max-w-[1920px] px-4 py-3 lg:px-6 lg:py-4">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <section class="flex flex-col gap-4">
      <div class="surface-card p-6">
        <div class="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div class="max-w-4xl">
            <p class="text-xs uppercase tracking-[0.2em] text-sky-600 dark:text-sky-300">System Settings</p>
            <h2 class="mt-2 text-2xl font-semibold text-slate-900 dark:text-white">系统设置</h2>
            <p class="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
              这里直接管理 Agent 大模型配置、Medical SAM3 与 LoRA/adapter 加载，以及一些会真实影响运行效果的必要参数：
              像素标定、上传限制和推理超时。保存后后端会自动刷新可热更新的运行时配置。
            </p>
          </div>

          <div class="flex flex-wrap gap-2">
            <button
              type="button"
              class="surface-button-secondary px-4 py-2.5 text-sm"
              :disabled="!isDirty || isSaving || isLoading"
              @click="handleReset"
            >
              恢复已保存配置
            </button>
            <button
              type="button"
              class="surface-button-primary px-4 py-2.5 text-sm"
              :disabled="!isDirty || isSaving || isLoading || !form"
              @click="handleSave"
            >
              {{ isSaving ? '保存中...' : '保存并重载' }}
            </button>
          </div>
        </div>
      </div>

      <div v-if="isLoading" class="surface-card p-6">
        <div class="space-y-4 animate-pulse">
          <div class="h-6 w-64 rounded bg-gray-100 dark:bg-slate-700" />
          <div class="grid gap-4 xl:grid-cols-4">
            <div v-for="index in 4" :key="index" class="h-28 rounded-2xl bg-gray-100 dark:bg-slate-800" />
          </div>
          <div class="h-96 rounded-3xl bg-gray-100 dark:bg-slate-800" />
        </div>
      </div>

      <section
        v-else-if="form && runtimeStatus"
        class="grid gap-4"
      >
        <div class="grid gap-4 xl:grid-cols-4">
          <article class="surface-card p-5">
            <p class="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">LLM</p>
            <h3 class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">
              {{ runtimeStatus.llmReady ? '已就绪' : '待补全配置' }}
            </h3>
            <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">{{ activeProfileDescription }}</p>
          </article>

          <article class="surface-card p-5">
            <p class="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">SAM3</p>
            <h3 class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ runtimeHeadline }}</h3>
            <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
              当前模式：{{ runtimeStatus.sam3RuntimeMode === 'mock' ? 'Mock' : 'SAM3' }}
            </p>
          </article>

          <article class="surface-card p-5">
            <p class="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">LoRA / Adapter</p>
            <h3 class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">
              {{ runtimeStatus.loraLoaded ? '已加载' : '未激活' }}
            </h3>
            <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
              仅在 SAM3 真实模式下生效，适合针对特定内镜场景加载附加权重。
            </p>
          </article>

          <article class="surface-card p-5">
            <p class="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">运行边界</p>
            <h3 class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">
              {{ form.runtime.inferenceTimeoutSeconds }}s / {{ form.runtime.maxUploadSizeMb }}MB
            </h3>
            <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
              推理超时和上传上限会直接作用于现有分析接口。
            </p>
          </article>
        </div>

        <div
          v-if="runtimeStatus.warnings.length"
          class="surface-card border border-amber-200 bg-amber-50 p-5 text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/50 dark:text-amber-100"
        >
          <h3 class="text-sm font-semibold">运行提示</h3>
          <p
            v-for="warning in runtimeStatus.warnings"
            :key="warning"
            class="mt-2 text-sm leading-6"
          >
            {{ warning }}
          </p>
        </div>

        <div class="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_420px]">
          <div class="grid gap-4">
            <section class="surface-card p-6">
              <div class="flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-800">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white">大模型 API 配置</h3>
                <p class="text-sm leading-6 text-slate-600 dark:text-slate-300">
                  这里改成了真正的 Profile 管理面板。可以创建、修改、删除多套 API 配置，并从样式化的卡片列表中切换活动 Profile。
                </p>
              </div>

              <div class="mt-5 flex flex-wrap gap-2">
                <button
                  type="button"
                  class="surface-button-primary px-4 py-2.5 text-sm"
                  @click="handleCreateProfile('openai_compatible')"
                >
                  新建 OpenAI Profile
                </button>
                <button
                  type="button"
                  class="surface-button-secondary px-4 py-2.5 text-sm"
                  @click="handleCreateProfile('modelscope')"
                >
                  新建 ModelScope Profile
                </button>
              </div>

              <div class="mt-5 grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
                <div class="grid content-start gap-3">
                  <button
                    v-for="profile in form.llm.profiles"
                    :key="profile.profileId"
                    type="button"
                    class="rounded-3xl border p-4 text-left transition"
                    :class="profile.profileId === form.llm.activeProfile
                      ? 'border-sky-400 bg-sky-50 shadow-soft dark:border-sky-500 dark:bg-sky-950/50'
                      : 'border-slate-200 bg-white hover:border-slate-300 dark:border-slate-800 dark:bg-slate-950 dark:hover:border-slate-700'"
                    @click="handleSetActiveProfile(profile.profileId)"
                  >
                    <div class="flex items-start justify-between gap-3">
                      <div>
                        <p class="text-sm font-semibold text-slate-900 dark:text-white">{{ profile.profileId }}</p>
                        <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">
                          {{ profile.providerKind === 'openai_compatible' ? 'OpenAI Compatible' : 'ModelScope' }}
                        </p>
                      </div>
                      <span
                        v-if="profile.profileId === form.llm.activeProfile"
                        class="rounded-full bg-sky-100 px-2.5 py-1 text-[11px] font-medium text-sky-700 dark:bg-sky-900/70 dark:text-sky-200"
                      >
                        活动中
                      </span>
                    </div>

                    <p class="mt-3 text-xs text-slate-500 dark:text-slate-400">{{ profile.defaultModel }}</p>

                    <div class="mt-4 flex gap-2">
                      <button
                        type="button"
                        class="rounded-full border border-rose-200 px-3 py-1 text-xs text-rose-600 transition hover:border-rose-300 hover:text-rose-700 dark:border-rose-900/70 dark:text-rose-300 dark:hover:border-rose-700"
                        @click.stop="handleDeleteProfile(profile.profileId)"
                      >
                        删除
                      </button>
                    </div>
                  </button>
                </div>

                <article
                  v-if="activeLlmProfile"
                  class="rounded-3xl border border-slate-200 p-5 dark:border-slate-800"
                >
                  <div class="flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-800 md:flex-row md:items-end md:justify-between">
                    <div>
                      <h4 class="text-base font-semibold text-slate-900 dark:text-white">活动 Profile 详情</h4>
                      <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
                        {{ activeProfileDescription }}
                      </p>
                    </div>
                    <span class="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      {{ activeProfileKindLabel }}
                    </span>
                  </div>

                  <div class="mt-4 grid gap-3 md:grid-cols-2">
                    <label>
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Profile 标识</span>
                      <input
                        :value="activeLlmProfile.profileId"
                        :class="textInputClass"
                        @change="handleRenameActiveProfile(($event.target as HTMLInputElement).value)"
                      >
                    </label>

                    <label>
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Provider 类型</span>
                      <div class="relative mt-2">
                        <select
                          :value="activeLlmProfile.providerKind"
                          :class="selectInputClass"
                          @change="handleProviderKindChange(($event.target as HTMLSelectElement).value)"
                        >
                          <option value="openai_compatible">OpenAI Compatible</option>
                          <option value="modelscope">ModelScope</option>
                        </select>
                        <span class="pointer-events-none absolute inset-y-0 right-4 flex items-center text-slate-400">▾</span>
                      </div>
                    </label>

                    <label>
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Provider</span>
                      <input v-model="activeLlmProfile.defaultProvider" :class="textInputClass">
                    </label>

                    <label>
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Model</span>
                      <input v-model="activeLlmProfile.defaultModel" :class="textInputClass">
                    </label>

                    <label class="md:col-span-2">
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">API Key</span>
                      <input v-model="activeLlmProfile.apiKey" type="password" :class="textInputClass">
                    </label>

                    <label class="md:col-span-2">
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Base URL</span>
                      <input v-model="activeLlmProfile.baseUrl" :class="textInputClass">
                    </label>

                    <label>
                      <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Timeout (s)</span>
                      <input v-model.number="activeLlmProfile.timeout" type="number" min="1" max="600" :class="textInputClass">
                    </label>
                  </div>
                </article>
              </div>
            </section>

            <section class="surface-card p-6">
              <div class="flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-800">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white">Medical SAM3 与 LoRA</h3>
                <p class="text-sm leading-6 text-slate-600 dark:text-slate-300">
                  这里控制分割模型的真实/Mock 模式、主 checkpoint、设备和可选的 LoRA/adapter 权重加载。
                </p>
              </div>

              <div class="mt-5 grid gap-4 md:grid-cols-2">
                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">运行模式</span>
                  <select
                    v-model="form.sam3.loadMode"
                    :class="selectInputClass"
                  >
                    <option value="mock">Mock 联调</option>
                    <option value="sam3">SAM3 真实推理</option>
                  </select>
                </label>

                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">设备</span>
                  <input v-model="form.sam3.device" :class="textInputClass">
                </label>

                <label class="md:col-span-2">
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">主 Checkpoint 路径</span>
                  <input v-model="form.sam3.checkpointPath" :class="textInputClass">
                </label>

                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">输入尺寸</span>
                  <input v-model.number="form.sam3.inputSize" type="number" min="256" max="4096" :class="textInputClass">
                </label>

                <div class="grid gap-3 md:grid-cols-2 md:col-span-2">
                  <label class="flex items-center gap-3 rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 dark:border-slate-800 dark:text-slate-200">
                    <input v-model="form.sam3.keepAspectRatio" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                    保持输入长宽比
                  </label>
                  <label class="flex items-center gap-3 rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 dark:border-slate-800 dark:text-slate-200">
                    <input v-model="form.sam3.warmupEnabled" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                    启动时预热模型
                  </label>
                  <label class="flex items-center gap-3 rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 dark:border-slate-800 dark:text-slate-200 md:col-span-2">
                    <input v-model="form.sam3.loraEnabled" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                    启用 LoRA / adapter 权重加载
                  </label>
                </div>

                <label class="md:col-span-2">
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">LoRA / Adapter 路径</span>
                  <input v-model="form.sam3.loraPath" :class="textInputClass">
                </label>
              </div>
            </section>
          </div>

          <div class="grid content-start gap-4">
            <section class="surface-card p-6">
              <div class="flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-800">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white">Agent 工作流</h3>
                <p class="text-sm leading-6 text-slate-600 dark:text-slate-300">
                  这里保留必要但会真实影响结果的策略项，不提供模块级开关。
                </p>
              </div>

              <div class="mt-5 grid gap-3">
                <label class="flex items-center gap-3 rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 dark:border-slate-800 dark:text-slate-200">
                  <input v-model="form.agent.enableLlm" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                  启用 LLM 推理增强
                </label>
                <label class="flex items-center gap-3 rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 dark:border-slate-800 dark:text-slate-200">
                  <input v-model="form.agent.enableLlmReport" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                  启用 LLM 报告生成
                </label>
                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">像素尺寸标定 (mm / px)</span>
                  <input v-model.number="form.agent.pixelSizeMm" type="number" min="0.01" max="10" step="0.01" :class="textInputClass">
                  <p class="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
                    这是我补进去的必要设置项之一，会直接影响病灶大小估计与后续风险判断。
                  </p>
                </label>
              </div>
            </section>

            <section class="surface-card p-6">
              <div class="flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-800">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white">运行安全与联调</h3>
                <p class="text-sm leading-6 text-slate-600 dark:text-slate-300">
                  这些设置不会改变算法本身，但会直接影响系统稳定性、演示体验和接口边界。
                </p>
              </div>

              <div class="mt-5 grid gap-3">
                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">推理超时 (s)</span>
                  <input v-model.number="form.runtime.inferenceTimeoutSeconds" type="number" min="1" max="300" :class="textInputClass">
                </label>
                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">上传大小限制 (MB)</span>
                  <input v-model.number="form.runtime.maxUploadSizeMb" type="number" min="1" max="200" :class="textInputClass">
                </label>
                <label>
                  <span class="text-xs font-medium text-slate-500 dark:text-slate-400">Mock 延迟 (ms)</span>
                  <input v-model.number="form.runtime.mockDelayMs" type="number" min="0" max="10000" :class="textInputClass">
                </label>
              </div>
            </section>

            <section class="surface-card p-6">
              <h3 class="text-lg font-semibold text-slate-900 dark:text-white">配置文件位置</h3>
              <div class="mt-4 grid gap-3 text-sm text-slate-600 dark:text-slate-300">
                <div class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
                  <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">LLM Config</p>
                  <p class="mt-2 break-all">{{ runtimeStatus.llmConfigPath }}</p>
                </div>
                <div class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
                  <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Runtime Overrides</p>
                  <p class="mt-2 break-all">{{ runtimeStatus.runtimeSettingsPath }}</p>
                </div>
              </div>
            </section>
          </div>
        </div>
      </section>

      <section v-else class="surface-card p-6">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">系统设置加载失败</h3>
        <p class="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
          {{ loadErrorMessage || '当前无法从后端读取设置，请确认 Backend 服务已启动。' }}
        </p>
        <button type="button" class="surface-button-primary mt-4 px-4 py-2.5 text-sm" @click="loadSettings">
          重新加载
        </button>
      </section>
    </section>
  </main>
</template>