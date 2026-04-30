<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import {
  fetchSmartAnnotationTags,
  getReportBuilderMockContext,
  invokeReportDraftAgent,
  saveReportDraft,
} from '@/api/reportBuilder'
import ReportGenerationWorkspace from '@/components/report/ReportGenerationWorkspace.vue'
import type {
  AgentWorkflowSummary,
  AnnotationTag,
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
const captureImages = ref<string[]>([])
const initialOpinion = ref('')
const findings = ref('')
const conclusion = ref('')
const layoutSuggestion = ref('')
const streamText = ref('')
const annotationTags = ref<AnnotationTag[]>([])
const agentWorkflow = ref<AgentWorkflowSummary | null>(null)
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
  captureImages.value = [...nextContext.captureImageSrcs]
  initialOpinion.value = nextContext.initialOpinion
  annotationTags.value = []
  agentWorkflow.value = null
  streamText.value = ''
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
    showToast('报告生成上下文加载失败。', 'error')
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
    agentWorkflow.value = response.workflow
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
    const response = await fetchSmartAnnotationTags(payload)
    annotationTags.value = response.tags
    agentWorkflow.value = response.workflow
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

const handlePatientEdit = (patientId: string) => {
  showToast(`已触发患者 ${patientId} 的编辑入口。`)
}

const handleViewHistory = () => {
  showToast('已触发患者历史记录查看。')
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
  <main class="mx-auto flex min-h-[calc(100vh-88px)] w-full max-w-[1920px] flex-col overflow-hidden px-4 py-3 lg:px-6 lg:py-4">
    <ReportGenerationWorkspace
      :context="context"
      :capture-images="captureImages"
      :initial-opinion="initialOpinion"
      :findings="findings"
      :conclusion="conclusion"
      :layout-suggestion="layoutSuggestion"
      :stream-text="streamText"
      :annotation-tags="annotationTags"
      :agent-workflow="agentWorkflow"
      :tags-loading="tagsLoading"
      :tag-error-message="tagErrorMessage"
      :is-hydrating="isHydrating"
      :is-agent-loading="isAgentLoading"
      :is-saving="isSaving"
      :can-save-draft="canSaveDraft"
      :formatted-saved-at="formattedSavedAt"
      :toast-visible="toastVisible"
      :toast-message="toastMessage"
      :toast-tone="toastTone"
      @invoke-agent="handleInvokeAgent"
      @save-draft="handleSaveDraft"
      @fetch-agent-tags="handleFetchAgentTags"
      @tag-click="handleTagClick"
      @patient-edit="handlePatientEdit"
      @view-history="handleViewHistory"
      @update:initial-opinion="initialOpinion = $event"
      @update:findings="findings = $event"
      @update:conclusion="conclusion = $event"
      @update:layout-suggestion="layoutSuggestion = $event"
    />
  </main>
</template>