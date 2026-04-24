<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import {
  fetchSmartAnnotationTags,
  getReportBuilderMockContext,
  invokeReportDraftAgent,
  saveReportDraft,
} from '@/api/reportBuilder'
import EndoVideoPlayer from '@/components/common/EndoVideoPlayer.vue'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import PatientInfoCard from '@/components/common/PatientInfoCard.vue'
import SmartAnnotationTags from '@/components/common/SmartAnnotationTags.vue'
import TumorMaskViewer from '@/components/common/TumorMaskViewer.vue'
import ReportAgentForm from '@/components/report/ReportAgentForm.vue'
import ReportCaptureGallery from '@/components/report/ReportCaptureGallery.vue'
import ReportDraftStatusCard from '@/components/report/ReportDraftStatusCard.vue'
import ReportPreviewPanel from '@/components/report/ReportPreviewPanel.vue'
import type {
  AnnotationTag,
  CaptureFramePayload,
  FetchAnnotationTagsRequest,
  GenerateReportDraftRequest,
  ReportContextData,
  SaveReportDraftRequest,
} from '@/types/eis'

const props = defineProps<{
  reportId?: string
  contextData?: ReportContextData
}>()

const emit = defineEmits<{
  (event: 'invoke-agent', payload: GenerateReportDraftRequest): void
  (event: 'save-draft', payload: SaveReportDraftRequest): void
}>()

const context = ref<ReportContextData | null>(null)
const playerIsPlaying = ref(false)
const showMask = ref(true)
const captureImages = ref<string[]>([])
const initialOpinion = ref('')
const findings = ref('')
const conclusion = ref('')
const layoutSuggestion = ref('')
const streamText = ref('')
const annotationTags = ref<AnnotationTag[]>([])
const tagsLoading = ref(false)
const tagErrorMessage = ref('')
const isHydrating = ref(false)
const isAgentLoading = ref(false)
const isSaving = ref(false)
const lastSavedAt = ref('')
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

const applyContext = (nextContext: ReportContextData) => {
  context.value = nextContext
  showMask.value = nextContext.showMask
  captureImages.value = [...nextContext.captureImageSrcs]
  initialOpinion.value = nextContext.initialOpinion
}

const hydrateContext = async () => {
  isHydrating.value = true

  try {
    if (props.contextData) {
      applyContext(props.contextData)
    } else {
      const mockContext = await getReportBuilderMockContext(props.reportId)
      applyContext(mockContext)
    }
  } catch {
    showToast('报告构建上下文加载失败。', 'error')
  } finally {
    isHydrating.value = false
  }
}

const draftRequest = computed<GenerateReportDraftRequest | null>(() => {
  if (!context.value) {
    return null
  }

  return {
    reportId: props.reportId,
    patientId: context.value.patient.patientId,
    contextData: {
      ...context.value,
      showMask: showMask.value,
      captureImageSrcs: captureImages.value,
      reportSnippet: initialOpinion.value || context.value.reportSnippet,
      initialOpinion: initialOpinion.value,
    },
  }
})

const saveDraftRequest = computed<SaveReportDraftRequest | null>(() => {
  if (!context.value) {
    return null
  }

  return {
    reportId: props.reportId,
    patientId: context.value.patient.patientId,
    findings: findings.value.trim(),
    conclusion: conclusion.value.trim(),
    layoutSuggestion: layoutSuggestion.value.trim(),
  }
})

const canSaveDraft = computed(() => {
  return Boolean(saveDraftRequest.value?.findings || saveDraftRequest.value?.conclusion)
})

const formattedSavedAt = computed(() => {
  if (!lastSavedAt.value) {
    return '尚未保存'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(lastSavedAt.value))
})

const handleInvokeAgent = async () => {
  if (!draftRequest.value) {
    return
  }

  streamText.value = ''
  isAgentLoading.value = true
  emit('invoke-agent', draftRequest.value)

  try {
    const response = await invokeReportDraftAgent(draftRequest.value, (chunk) => {
      streamText.value += chunk
    })

    findings.value = response.findings
    conclusion.value = response.conclusion
    layoutSuggestion.value = response.layoutSuggestion
    showToast('Agent 已生成结构化草稿。', 'success')
  } catch {
    showToast('Agent 草稿生成失败，请稍后重试。', 'error')
  } finally {
    isAgentLoading.value = false
  }
}

const handleSaveDraft = async () => {
  if (!saveDraftRequest.value) {
    return
  }

  isSaving.value = true
  emit('save-draft', saveDraftRequest.value)

  try {
    const savedDraft = await saveReportDraft(saveDraftRequest.value)
    lastSavedAt.value = savedDraft.updatedAt
    showToast('报告草稿已保存。', 'success')
  } catch {
    showToast('草稿保存失败，请检查网络或稍后重试。', 'error')
  } finally {
    isSaving.value = false
  }
}

const handleFetchAgentTags = async (payload: FetchAnnotationTagsRequest) => {
  tagsLoading.value = true
  tagErrorMessage.value = ''

  try {
    annotationTags.value = await fetchSmartAnnotationTags(payload)
  } catch {
    annotationTags.value = []
    tagErrorMessage.value = '标签分析服务暂不可用。'
    showToast('标签分析失败。', 'error')
  } finally {
    tagsLoading.value = false
  }
}

const handleTagClick = (tag: AnnotationTag) => {
  showToast(`已定位标签 ${tag.label}，建议回看 ${tag.targetTime.toFixed(1)} 秒。`)
}

const handleCaptureFrame = (payload: CaptureFramePayload) => {
  captureImages.value = [payload.dataUrl, ...captureImages.value].slice(0, 6)
  showToast(payload.includesMask ? '已抓取带遮罩帧。' : '已抓取原始帧。', 'success')
}

const handlePatientEdit = (patientId: string) => {
  showToast(`已触发患者 ${patientId} 的编辑入口。`)
}

const handleViewHistory = () => {
  showToast('已触发患者历史记录查看。')
}

const handleExpandTumorView = () => {
  showToast('已预留全屏查看入口，可对接 Dialog 或工作站。')
}

watch(
  () => props.contextData,
  (nextValue) => {
    if (nextValue) {
      applyContext(nextValue)
    }
  },
  { deep: true },
)

onMounted(async () => {
  await hydrateContext()
})
</script>

<template>
  <section class="relative space-y-6">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h2 class="text-2xl font-semibold text-gray-900 dark:text-white">ReportBuilderView</h2>
        <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
          页面层负责调 Agent Mock API、管理状态并将结果分发给哑组件。右侧预览保持 A4 报告纸布局。
        </p>
      </div>
      <div class="flex flex-wrap gap-3">
        <button
          type="button"
          class="surface-button-primary px-5 py-3"
          :disabled="isHydrating || isAgentLoading || !context"
          @click="handleInvokeAgent"
        >
          {{ isAgentLoading ? '生成中...' : '智能生成草稿' }}
        </button>
        <button
          type="button"
          class="surface-button-secondary px-5 py-3"
          :disabled="isSaving || !canSaveDraft"
          @click="handleSaveDraft"
        >
          {{ isSaving ? '保存中...' : '保存草稿' }}
        </button>
      </div>
    </div>

    <div v-if="isHydrating && !context" class="surface-card p-8">
      <div class="space-y-4 animate-pulse">
        <div class="h-6 w-48 rounded bg-gray-100 dark:bg-slate-700" />
        <div class="h-64 rounded-2xl bg-gray-100 dark:bg-slate-800" />
        <div class="h-48 rounded-2xl bg-gray-100 dark:bg-slate-800" />
      </div>
    </div>

    <div v-else-if="context" class="grid gap-8 xl:grid-cols-[minmax(0,1.25fr)_420px]">
      <div class="space-y-6">
        <PatientInfoCard
          :patient-name="context.patient.patientName"
          :gender="context.patient.gender"
          :age="context.patient.age"
          :patient-id="context.patient.patientId"
          :exam-date="context.patient.examDate"
          :status="context.patient.status"
          @edit="handlePatientEdit"
          @view-history="handleViewHistory"
        />

        <EndoVideoPlayer
          :video-src="context.videoSrc"
          :is-playing="playerIsPlaying"
          :mask-data="context.maskData"
          :show-mask="showMask"
          @capture-frame="handleCaptureFrame"
          @play-state-change="playerIsPlaying = $event.isPlaying"
          @update:show-mask="showMask = $event"
        />

        <ReportCaptureGallery :images="captureImages" />

        <SmartAnnotationTags
          :video-frame-data="context.videoFrameData"
          :report-snippet="initialOpinion"
          :tags="annotationTags"
          :is-loading="tagsLoading"
          :error-message="tagErrorMessage"
          @fetch-agent-tags="handleFetchAgentTags"
          @tag-click="handleTagClick"
        />

        <ReportAgentForm
          :initial-opinion="initialOpinion"
          :findings="findings"
          :conclusion="conclusion"
          :layout-suggestion="layoutSuggestion"
          :stream-text="streamText"
          :is-agent-loading="isAgentLoading"
          @update:initial-opinion="initialOpinion = $event"
          @update:findings="findings = $event"
          @update:conclusion="conclusion = $event"
          @update:layout-suggestion="layoutSuggestion = $event"
        />

        <TumorMaskViewer
          :tumor-image-src="context.tumorFocus.tumorImageSrc"
          :mask-data="context.tumorFocus.maskData"
          :details="context.tumorFocus.details"
          @expand-view="handleExpandTumorView"
          @toggle-mask="showToast($event ? '局部遮罩已开启。' : '局部遮罩已关闭。')"
        />
      </div>

      <div class="space-y-6 xl:sticky xl:top-6 xl:self-start">
        <ReportDraftStatusCard :saved-at-label="formattedSavedAt" :annotation-count="annotationTags.length" />

        <ReportPreviewPanel
          class="min-h-[640px]"
          :patient="context.patient"
          :saved-at-label="formattedSavedAt"
          :findings="findings"
          :conclusion="conclusion"
          :layout-suggestion="layoutSuggestion"
          :annotation-count="annotationTags.length"
        />
      </div>
    </div>
  </section>
</template>