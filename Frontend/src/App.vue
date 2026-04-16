<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { getPatientPreviewCards } from '@/api/reportBuilder'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import PatientInfoCard from '@/components/common/PatientInfoCard.vue'
import ThemeToggleButton from '@/components/common/ThemeToggleButton.vue'
import { useThemeStore } from '@/stores/theme'
import type { GenerateReportDraftRequest, PatientRecord, SaveReportDraftRequest } from '@/types/eis'
import ReportBuilderView from '@/views/ReportBuilderView.vue'

const themeStore = useThemeStore()
const previewPatients = ref<PatientRecord[]>([])
const toastVisible = ref(false)
const toastMessage = ref('')
const toastTone = ref<'info' | 'success' | 'error'>('info')

let toastTimer: number | undefined

const showToast = (message: string, tone: 'info' | 'success' | 'error' = 'info') => {
  toastMessage.value = message
  toastTone.value = tone
  toastVisible.value = true

  if (toastTimer) {
    window.clearTimeout(toastTimer)
  }

  toastTimer = window.setTimeout(() => {
    toastVisible.value = false
  }, 2600)
}

const loadPreviewPatients = async () => {
  try {
    previewPatients.value = await getPatientPreviewCards()
  } catch {
    showToast('患者预览数据加载失败。', 'error')
  }
}

const handlePreviewEdit = (patientId: string) => {
  showToast(`已触发患者 ${patientId} 的编辑动作。`)
}

const handlePreviewHistory = (patientId: string) => {
  showToast(`已触发患者 ${patientId} 的历史记录查看。`)
}

const handleInvokeAgent = (request: GenerateReportDraftRequest) => {
  showToast(`已向 Agent 发起 ${request.patientId} 的报告草稿生成请求。`)
}

const handleSaveDraft = (request: SaveReportDraftRequest) => {
  showToast(`已触发 ${request.patientId} 的草稿保存动作。`, 'success')
}

onMounted(async () => {
  themeStore.initializeTheme()
  await loadPreviewPatients()
})
</script>

<template>
  <div class="app-shell">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <div class="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-6 py-6 lg:px-8 lg:py-8">
      <header class="surface-card overflow-hidden p-6 lg:p-8">
        <div class="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div class="max-w-3xl space-y-4">
            <span class="surface-badge bg-blue-50 text-blue-700 dark:bg-sky-950/70 dark:text-sky-200">
              Vue 3 Component-First / EIS Demo
            </span>
            <div class="space-y-3">
              <h1 class="text-3xl font-semibold tracking-tight text-gray-900 dark:text-white lg:text-4xl">
                内镜检查报告构建工作台
              </h1>
              <p class="max-w-2xl text-sm leading-7 text-gray-500 dark:text-gray-400 lg:text-base">
                该演示优先固化患者信息卡、内镜播放器、智能标签和肿瘤 ROI 组件，并通过 Mock Agent
                API 在页面层完成报告草稿生成、保存和预览。
              </p>
            </div>
          </div>

          <ThemeToggleButton :is-dark="themeStore.isDark" :mode="themeStore.mode" @toggle="themeStore.toggleTheme" />
        </div>
      </header>

      <section class="space-y-4">
        <div class="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">PatientInfoCard 预览</h2>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              使用 snake_case Mock 数据映射为 camelCase 后渲染 3 条患者基本信息。
            </p>
          </div>
        </div>

        <div v-if="previewPatients.length" class="grid gap-6 xl:grid-cols-3">
          <PatientInfoCard
            v-for="patient in previewPatients"
            :key="patient.patientId"
            :patient-name="patient.patientName"
            :gender="patient.gender"
            :age="patient.age"
            :patient-id="patient.patientId"
            :exam-date="patient.examDate"
            :status="patient.status"
            @edit="handlePreviewEdit"
            @view-history="handlePreviewHistory(patient.patientId)"
          />
        </div>

        <div v-else class="grid gap-6 xl:grid-cols-3">
          <div v-for="index in 3" :key="index" class="surface-card h-64 animate-pulse bg-gray-100 dark:bg-slate-800" />
        </div>
      </section>

      <ReportBuilderView @invoke-agent="handleInvokeAgent" @save-draft="handleSaveDraft" />
    </div>
  </div>
</template>
