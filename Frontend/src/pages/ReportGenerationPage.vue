<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import {
  fetchSmartAnnotationTags,
  getReportBuilderContext,
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

/**
 * brief:
 *   Handle props.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const props = defineProps<{
  reportId?: string
  contextData?: ReportContextData
}>()

/**
 * brief:
 *   Handle emit.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const emit = defineEmits<{
  (event: 'invoke-agent', payload: GenerateReportDraftRequest): void
  (event: 'save-draft', payload: SaveReportDraftRequest): void
}>()

/**
 * brief:
 *   Handle context.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const context = ref<ReportContextData | null>(null)
/**
 * brief:
 *   Handle capture images.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const captureImages = ref<string[]>([])
/**
 * brief:
 *   Handle initial opinion.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const initialOpinion = ref('')
/**
 * brief:
 *   Handle findings.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const findings = ref('')
/**
 * brief:
 *   Handle conclusion.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const conclusion = ref('')
/**
 * brief:
 *   Handle layout suggestion.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const layoutSuggestion = ref('')
/**
 * brief:
 *   Handle stream text.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const streamText = ref('')
/**
 * brief:
 *   Handle annotation tags.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const annotationTags = ref<AnnotationTag[]>([])
/**
 * brief:
 *   Handle agent workflow.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const agentWorkflow = ref<AgentWorkflowSummary | null>(null)
/**
 * brief:
 *   Handle tags loading.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const tagsLoading = ref(false)
/**
 * brief:
 *   Handle tag error message.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const tagErrorMessage = ref('')
/**
 * brief:
 *   Handle is hydrating.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const isHydrating = ref(false)
/**
 * brief:
 *   Handle is agent loading.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const isAgentLoading = ref(false)
/**
 * brief:
 *   Handle is saving.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const isSaving = ref(false)
/**
 * brief:
 *   Handle last saved at.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const lastSavedAt = ref('')
/**
 * brief:
 *   Handle toast visible.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const toastVisible = ref(false)
/**
 * brief:
 *   Handle toast message.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const toastMessage = ref('')
/**
 * brief:
 *   Handle toast tone.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const toastTone = ref<'info' | 'success' | 'error'>('info')

let toastTimer: number | undefined

/**
 * brief:
 *   Handle show toast.
 *
 * parameter:
 *   - message: Input value for message.
 *   - tone: Input value for tone.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Apply context.
 *
 * parameter:
 *   - nextContext: Input value for nextContext.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const applyContext = (nextContext: ReportContextData) => {
  context.value = nextContext
  captureImages.value = [...nextContext.captureImageSrcs]
  initialOpinion.value = nextContext.initialOpinion
  annotationTags.value = []
  agentWorkflow.value = null
  streamText.value = ''
}

/**
 * brief:
 *   Handle hydrate context.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const hydrateContext = async () => {
  isHydrating.value = true

  try {
    if (props.contextData) {
      applyContext(props.contextData)
    } else {
      /**
       * brief:
       *   Handle remote context.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const remoteContext = await getReportBuilderContext(props.reportId)
      applyContext(remoteContext)
    }
  } catch {
    showToast('报告生成上下文加载失败。', 'error')
  } finally {
    isHydrating.value = false
  }
}

/**
 * brief:
 *   Handle draft request.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Save draft request.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle can save draft.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const canSaveDraft = computed(() => {
  return Boolean(saveDraftRequest.value?.findings || saveDraftRequest.value?.conclusion)
})

/**
 * brief:
 *   Handle formatted saved at.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle invoke agent.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handleInvokeAgent = async () => {
  if (!draftRequest.value) {
    return
  }

  streamText.value = ''
  isAgentLoading.value = true
  emit('invoke-agent', draftRequest.value)

  try {
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

/**
 * brief:
 *   Handle save draft.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handleSaveDraft = async () => {
  if (!saveDraftRequest.value) {
    return
  }

  isSaving.value = true
  emit('save-draft', saveDraftRequest.value)

  try {
    /**
     * brief:
     *   Handle saved draft.
     *
     * parameter:
     *   - payload: Input value for payload.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const savedDraft = await saveReportDraft(saveDraftRequest.value)
    lastSavedAt.value = savedDraft.updatedAt
    showToast('报告草稿已保存。', 'success')
  } catch {
    showToast('草稿保存失败，请检查网络或稍后重试。', 'error')
  } finally {
    isSaving.value = false
  }
}

/**
 * brief:
 *   Handle fetch agent tags.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handleFetchAgentTags = async (payload: FetchAnnotationTagsRequest) => {
  tagsLoading.value = true
  tagErrorMessage.value = ''

  try {
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

/**
 * brief:
 *   Handle tag click.
 *
 * parameter:
 *   - tag: Input value for tag.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handleTagClick = (tag: AnnotationTag) => {
  showToast(`已定位标签 ${tag.label}，建议回看 ${tag.targetTime.toFixed(1)} 秒。`)
}

/**
 * brief:
 *   Handle patient edit.
 *
 * parameter:
 *   - patientId: Input value for patientId.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handlePatientEdit = (patientId: string) => {
  showToast(`已触发患者 ${patientId} 的编辑入口。`)
}

/**
 * brief:
 *   Handle view history.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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
