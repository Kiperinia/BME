<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { getSystemSettings, updateSystemSettings } from '@/api/systemSettings'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import type { LlmProfile, LlmProviderKind, SystemSettingsPayload, SystemSettingsStatus } from '@/types/systemSettings'

type ProfileCreationKind = LlmProviderKind | 'deepseek'

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

const cloneSettings = (value: SystemSettingsPayload): SystemSettingsPayload => {
  return JSON.parse(JSON.stringify(value)) as SystemSettingsPayload
}

const createProfileDraft = (providerKind: ProfileCreationKind, profileId: string): LlmProfile => {
  if (providerKind === 'deepseek') {
    return {
      profileId,
      providerKind: 'openai_compatible',
      defaultProvider: 'deepseek',
      defaultModel: 'deepseek-chat',
      apiKey: '',
      baseUrl: 'https://api.deepseek.com/v1',
      timeout: 60,
    }
  }

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

const buildUniqueProfileId = (baseId: string, existingIds: string[]) => {
  let candidate = baseId
  let suffix = 2

  while (existingIds.includes(candidate)) {
    candidate = `${baseId}-${suffix}`
    suffix += 1
  }

  return candidate
}

const pushToast = (message: string, tone: 'info' | 'success' | 'error' = 'info') => {
  toastVisible.value = true
  toastMessage.value = message
  toastTone.value = tone

  if (toastTimer) {
    window.clearTimeout(toastTimer)
  }

  toastTimer = window.setTimeout(() => {
    toastVisible.value = false
  }, 2600)
}

const activeProfile = computed<LlmProfile | null>(() => {
  if (!form.value) {
    return null
  }

  return form.value.llm.profiles.find((profile) => profile.profileId === form.value?.llm.activeProfile) ?? null
})

const isDirty = computed(() => {
  if (!savedSettings.value || !form.value) {
    return false
  }

  return JSON.stringify(savedSettings.value) !== JSON.stringify(form.value)
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
    loadErrorMessage.value = error instanceof Error ? error.message : '系统设置加载失败。'
  } finally {
    isLoading.value = false
  }
}

const handleProviderKindChange = (value: LlmProviderKind) => {
  if (!activeProfile.value) {
    return
  }

  activeProfile.value.providerKind = value
  activeProfile.value.defaultProvider = value === 'modelscope' ? 'modelscope' : 'openai'
  if (value === 'modelscope' && !activeProfile.value.baseUrl.trim()) {
    activeProfile.value.baseUrl = 'https://api-inference.modelscope.cn/v1/'
  }
}

const handleCreateProfile = (providerKind: ProfileCreationKind) => {
  if (!form.value) {
    return
  }

  const baseId = providerKind === 'modelscope' ? 'modelscope-profile' : providerKind === 'deepseek' ? 'deepseek-profile' : 'openai-profile'
  const profileId = buildUniqueProfileId(
    baseId,
    form.value.llm.profiles.map((profile) => profile.profileId),
  )
  const nextProfile = createProfileDraft(providerKind, profileId)

  form.value.llm.profiles.push(nextProfile)
  form.value.llm.activeProfile = nextProfile.profileId
  pushToast(`已新增 ${providerKind === 'modelscope' ? 'ModelScope' : providerKind === 'deepseek' ? 'DeepSeek' : 'OpenAI'} API Profile。`, 'success')
}

const handleReset = () => {
  if (!savedSettings.value) {
    return
  }

  form.value = cloneSettings(savedSettings.value)
  pushToast('已恢复为当前已保存配置。')
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
    pushToast('系统设置已保存。', 'success')
  } catch (error) {
    pushToast(error instanceof Error ? error.message : '系统设置保存失败。', 'error')
  } finally {
    isSaving.value = false
  }
}

onMounted(async () => {
  await loadSettings()
})
</script>

<template>
  <main class="mx-auto flex min-h-[calc(100vh-88px)] w-full max-w-[1500px] flex-col gap-4 px-4 py-4 lg:px-6">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <section class="surface-card p-6">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div class="max-w-3xl">
          <p class="text-xs uppercase tracking-[0.2em] text-sky-600 dark:text-sky-300">Settings</p>
          <h2 class="mt-2 text-2xl font-semibold text-slate-900 dark:text-white">运行模式与 Agent API 配置</h2>
          <p class="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
            这里只保留工作台需要的两类设置：MedicalSAM3 的 Mock / SAM3 模式切换，以及 Agent 使用的 API Profile。
          </p>
        </div>

        <div class="flex flex-wrap gap-3">
          <button type="button" class="surface-button-secondary px-4 py-3" :disabled="!isDirty || isSaving" @click="handleReset">
            恢复已保存
          </button>
          <button type="button" class="surface-button-primary px-4 py-3" :disabled="!isDirty || isSaving || !form" @click="handleSave">
            {{ isSaving ? '保存中...' : '保存设置' }}
          </button>
        </div>
      </div>
    </section>

    <section v-if="isLoading" class="surface-card p-6">
      <div class="space-y-4 animate-pulse">
        <div class="h-8 w-64 rounded bg-slate-100 dark:bg-slate-800" />
        <div class="h-40 rounded-3xl bg-slate-100 dark:bg-slate-800" />
        <div class="h-64 rounded-3xl bg-slate-100 dark:bg-slate-800" />
      </div>
    </section>

    <section v-else-if="form && runtimeStatus && activeProfile" class="grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <article class="surface-card p-6">
        <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">MedicalSAM3 运行模式</h3>
          <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">切换前端联调使用的 Mock 模式或实际 SAM3 推理模式。</p>
        </div>

        <div class="mt-5 grid gap-4">
          <label>
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">运行模式</span>
            <select v-model="form.sam3.loadMode" class="surface-input mt-2">
              <option value="mock">Mock</option>
              <option value="sam3">SAM3</option>
            </select>
          </label>

          <label class="flex items-center gap-3 rounded-3xl border border-slate-200 px-4 py-3 dark:border-slate-700">
            <input v-model="form.sam3.loraEnabled" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
            <span class="text-sm font-medium text-slate-700 dark:text-slate-200">启用 LoRA 推理</span>
          </label>

          <div v-if="form.sam3.loraEnabled" class="grid gap-4 rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">LoRA Checkpoint</span>
              <input v-model="form.sam3.loraPath" class="surface-input mt-2">
            </label>

            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">LoRA Stage</span>
              <select v-model="form.sam3.loraStage" class="surface-input mt-2">
                <option value="stage_a">stage_a</option>
                <option value="stage_b">stage_b</option>
                <option value="stage_c">stage_c</option>
              </select>
            </label>
          </div>

          <div class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
            <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">当前状态</p>
            <p class="mt-2 text-sm font-medium text-slate-900 dark:text-white">
              {{ runtimeStatus.sam3Ready ? 'SAM3 Runtime Ready' : '等待运行时就绪' }}
            </p>
            <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
              当前模式为 {{ runtimeStatus.sam3RuntimeMode }}，LLM 就绪状态为 {{ runtimeStatus.llmReady ? 'ready' : 'not ready' }}。
            </p>
          </div>

          <div v-if="runtimeStatus.warnings.length" class="grid gap-2">
            <p
              v-for="warning in runtimeStatus.warnings"
              :key="warning"
              class="rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200"
            >
              {{ warning }}
            </p>
          </div>
        </div>
      </article>

      <article class="surface-card p-6">
        <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">Agent API Profile</h3>
          <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
            选择当前生效的 Profile，并编辑其 Provider、Model、Base URL、API Key 等必要字段。
          </p>
        </div>

        <div class="mt-5 grid gap-4">
          <div class="flex flex-wrap gap-3">
            <button type="button" class="surface-button-primary px-4 py-2.5 text-sm" @click="handleCreateProfile('openai_compatible')">
              新增 OpenAI Profile
            </button>
            <button type="button" class="surface-button-primary px-4 py-2.5 text-sm" @click="handleCreateProfile('deepseek')">
              新增 DeepSeek Profile
            </button>
            <button type="button" class="surface-button-secondary px-4 py-2.5 text-sm" @click="handleCreateProfile('modelscope')">
              新增 ModelScope Profile
            </button>
          </div>

          <label>
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Active Profile</span>
            <select v-model="form.llm.activeProfile" class="surface-input mt-2">
              <option v-for="profile in form.llm.profiles" :key="profile.profileId" :value="profile.profileId">
                {{ profile.profileId }}
              </option>
            </select>
          </label>

          <div class="grid gap-4 md:grid-cols-2">
            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Provider Kind</span>
              <select
                :value="activeProfile.providerKind"
                class="surface-input mt-2"
                @change="handleProviderKindChange(($event.target as HTMLSelectElement).value as LlmProviderKind)"
              >
                <option value="openai_compatible">OpenAI Compatible</option>
                <option value="modelscope">ModelScope</option>
              </select>
            </label>

            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Provider</span>
              <input v-model="activeProfile.defaultProvider" class="surface-input mt-2">
            </label>

            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Model</span>
              <input v-model="activeProfile.defaultModel" class="surface-input mt-2">
            </label>

            <label>
              <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Timeout (s)</span>
              <input v-model.number="activeProfile.timeout" type="number" min="1" max="600" class="surface-input mt-2">
            </label>
          </div>

          <label>
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Base URL</span>
            <input v-model="activeProfile.baseUrl" class="surface-input mt-2">
          </label>

          <label>
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">API Key</span>
            <input v-model="activeProfile.apiKey" type="password" class="surface-input mt-2">
          </label>
        </div>
      </article>
    </section>

    <section v-if="form && runtimeStatus" class="surface-card p-6">
      <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">报告生成工作流配置</h3>
        <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
          配置 Agent 报告生成的工作流模式和反思优化参数。
        </p>
      </div>

      <div class="mt-5 grid gap-4 lg:grid-cols-2">
        <div class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
          <h4 class="text-sm font-medium text-slate-900 dark:text-white">报告生成模式</h4>
          <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">选择报告生成的方式</p>
          
          <label class="mt-3 flex items-center gap-3">
            <input 
              :checked="form.agent.useLlmReport" 
              @change="form.agent.useLlmReport = !form.agent.useLlmReport"
              type="checkbox" 
              class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
            <span class="text-sm font-medium text-slate-700 dark:text-slate-200">使用 LLM 生成自然语言报告</span>
          </label>
          <p class="mt-1 ml-7 text-xs text-slate-500 dark:text-slate-400">
            启用则使用 LLM 生成自然语言；禁用则使用规则引擎快速合成（0.3s）
          </p>

          <label class="mt-4 flex items-center gap-3">
            <input 
              :checked="form.agent.enableReportReflection" 
              @change="form.agent.enableReportReflection = !form.agent.enableReportReflection"
              type="checkbox" 
              class="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
            <span class="text-sm font-medium text-slate-700 dark:text-slate-200">启用 ReAct 反思优化</span>
          </label>
          <p class="mt-1 ml-7 text-xs text-slate-500 dark:text-slate-400">
            启用则进行自动分析、精修和评分优化（增加 25-30s）
          </p>
        </div>

        <div v-if="form?.agent?.enableReportReflection" class="rounded-3xl border border-sky-200 bg-sky-50 p-4 dark:border-sky-900/60 dark:bg-sky-950/30">
          <h4 class="text-sm font-medium text-sky-900 dark:text-sky-100">反思优化参数</h4>
          
          <label class="mt-3 block">
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-sky-700 dark:text-sky-300">
              最大迭代次数
            </span>
            <input 
              v-model.number="form.agent.reflectionMaxIterations" 
              type="number" 
              min="1" 
              max="10"
              class="surface-input mt-2"
            >
            <p class="mt-1 text-xs text-sky-600 dark:text-sky-400">
              单次诊断最多进行 {{ form.agent.reflectionMaxIterations }} 次智能优化迭代（建议 3）
            </p>
          </label>

          <label class="mt-3 block">
            <span class="text-xs font-medium uppercase tracking-[0.12em] text-sky-700 dark:text-sky-300">
              质量满足度阈值
            </span>
            <div class="flex items-center gap-3 mt-2">
              <input 
                v-model.number="form.agent.reflectionQualityThreshold" 
                type="range" 
                min="0" 
                max="10"
                step="0.1"
                class="flex-1"
              >
              <span class="text-sm font-medium text-sky-900 dark:text-sky-100 w-8 text-center">
                {{ form.agent.reflectionQualityThreshold.toFixed(1) }}
              </span>
            </div>
            <p class="mt-1 text-xs text-sky-600 dark:text-sky-400">
              当报告质量评分达到 {{ form.agent.reflectionQualityThreshold.toFixed(1) }}/10 时停止优化（建议 8.0）
            </p>
          </label>
        </div>

        <div v-else class="rounded-3xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900">
          <p class="text-sm text-slate-600 dark:text-slate-400">
            启用 ReAct 反思优化后，可在此配置迭代次数和质量阈值。
          </p>
        </div>
      </div>

      <div class="mt-4 rounded-3xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/60 dark:bg-amber-950/30">
        <p class="text-xs font-medium uppercase tracking-[0.14em] text-amber-700 dark:text-amber-300">工作流耗时预估</p>
        <p class="mt-2 text-sm text-amber-800 dark:text-amber-200">
          {{ form.agent.useLlmReport ? 'LLM ' : '模板 ' }}
          + {{ form.agent.enableReportReflection ? 'ReAct 反思' : '无反思' }}
          = 
          {{ form.agent.useLlmReport && form.agent.enableReportReflection ? '~35s' : form.agent.useLlmReport ? '~5s' : form.agent.enableReportReflection ? '~25s' : '~0.3s' }}
        </p>
      </div>
    </section>

    <section v-else class="surface-card p-6">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">系统设置加载失败</h3>
      <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
        {{ loadErrorMessage || '无法从后端读取当前设置。' }}
      </p>
      <button type="button" class="surface-button-primary mt-4 px-4 py-3" @click="loadSettings">
        重新加载
      </button>
    </section>
  </main>
</template>
