<script setup lang="ts">
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import PatientInfoCard from '@/components/common/PatientInfoCard.vue'
import ResizablePaneGroup from '@/components/common/ResizablePaneGroup.vue'
import SmartAnnotationTags from '@/components/common/SmartAnnotationTags.vue'
import ReportAgentForm from '@/components/report/ReportAgentForm.vue'
import ReportAgentWorkflowPanel from '@/components/report/ReportAgentWorkflowPanel.vue'
import ReportCaptureGallery from '@/components/report/ReportCaptureGallery.vue'
import ReportDraftStatusCard from '@/components/report/ReportDraftStatusCard.vue'
import ReportPreviewPanel from '@/components/report/ReportPreviewPanel.vue'
import type {
  AgentWorkflowSummary,
  AnnotationTag,
  FetchAnnotationTagsRequest,
  ReportContextData,
} from '@/types/eis'

defineProps<{
  context: ReportContextData | null
  captureImages: string[]
  initialOpinion: string
  findings: string
  conclusion: string
  layoutSuggestion: string
  streamText: string
  annotationTags: AnnotationTag[]
  agentWorkflow: AgentWorkflowSummary | null
  tagsLoading: boolean
  tagErrorMessage: string
  isHydrating: boolean
  isAgentLoading: boolean
  isSaving: boolean
  canSaveDraft: boolean
  formattedSavedAt: string
  toastVisible: boolean
  toastMessage: string
  toastTone: 'info' | 'success' | 'error'
}>()

const emit = defineEmits<{
  (event: 'invoke-agent'): void
  (event: 'save-draft'): void
  (event: 'fetch-agent-tags', payload: FetchAnnotationTagsRequest): void
  (event: 'tag-click', payload: AnnotationTag): void
  (event: 'patient-edit', patientId: string): void
  (event: 'view-history'): void
  (event: 'update:initial-opinion', value: string): void
  (event: 'update:findings', value: string): void
  (event: 'update:conclusion', value: string): void
  (event: 'update:layout-suggestion', value: string): void
}>()
</script>

<template>
  <section class="relative flex h-full min-h-0 flex-col gap-4">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <div class="flex flex-none flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h2 class="text-xl font-semibold text-gray-900 dark:text-white">智能报告生成</h2>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 lg:text-sm">
          单独承载标签分析、草稿生成、医生修订与报告预览，避免与视频检查流程混在同一页。
        </p>
      </div>
      <div class="flex flex-wrap gap-2">
        <button
          type="button"
          class="surface-button-primary px-4 py-2.5 text-sm"
          :disabled="isHydrating || isAgentLoading || !context"
          @click="emit('invoke-agent')"
        >
          {{ isAgentLoading ? '生成中...' : '智能生成草稿' }}
        </button>
        <button
          type="button"
          class="surface-button-secondary px-4 py-2.5 text-sm"
          :disabled="isSaving || !canSaveDraft"
          @click="emit('save-draft')"
        >
          {{ isSaving ? '保存中...' : '保存草稿' }}
        </button>
      </div>
    </div>

    <div v-if="isHydrating && !context" class="surface-card p-6">
      <div class="space-y-4 animate-pulse">
        <div class="h-6 w-48 rounded bg-gray-100 dark:bg-slate-700" />
        <div class="h-48 rounded-2xl bg-gray-100 dark:bg-slate-800" />
        <div class="h-72 rounded-2xl bg-gray-100 dark:bg-slate-800" />
      </div>
    </div>

    <ResizablePaneGroup
      v-else-if="context"
      storage-key="report-generation:main-layout"
      class="min-h-0 flex-1"
      orientation="horizontal"
      :pane-ids="['left', 'center', 'right']"
      :default-sizes="[22, 43, 35]"
      :min-sizes="[16, 24, 20]"
      :collapse-below="1440"
    >
      <template #left>
        <ResizablePaneGroup
          storage-key="report-generation:left-column"
          class="h-full min-h-0"
          orientation="vertical"
          :pane-ids="['patient', 'capture']"
          :default-sizes="[34, 66]"
          :min-sizes="[20, 20]"
        >
          <template #patient>
            <PatientInfoCard
              class="h-full min-h-0"
              :patient-name="context.patient.patientName"
              :gender="context.patient.gender"
              :age="context.patient.age"
              :patient-id="context.patient.patientId"
              :exam-date="context.patient.examDate"
              :status="context.patient.status"
              @edit="emit('patient-edit', $event)"
              @view-history="emit('view-history')"
            />
          </template>

          <template #capture>
            <ReportCaptureGallery class="h-full min-h-0" :images="captureImages" />
          </template>
        </ResizablePaneGroup>
      </template>

      <template #center>
        <ResizablePaneGroup
          storage-key="report-generation:center-column"
          class="h-full min-h-0"
          orientation="vertical"
          :pane-ids="['tags', 'workflow', 'form']"
          :default-sizes="[18, 30, 52]"
          :min-sizes="[14, 18, 24]"
        >
          <template #tags>
            <SmartAnnotationTags
              class="h-full min-h-0"
              :context-data="context"
              :report-snippet="initialOpinion"
              :tags="annotationTags"
              :is-loading="tagsLoading"
              :error-message="tagErrorMessage"
              @fetch-agent-tags="emit('fetch-agent-tags', $event)"
              @tag-click="emit('tag-click', $event)"
            />
          </template>

          <template #workflow>
            <ReportAgentWorkflowPanel class="h-full min-h-0" :workflow="agentWorkflow" />
          </template>

          <template #form>
            <ReportAgentForm
              class="h-full min-h-0"
              :initial-opinion="initialOpinion"
              :findings="findings"
              :conclusion="conclusion"
              :layout-suggestion="layoutSuggestion"
              :stream-text="streamText"
              :is-agent-loading="isAgentLoading"
              @update:initial-opinion="emit('update:initial-opinion', $event)"
              @update:findings="emit('update:findings', $event)"
              @update:conclusion="emit('update:conclusion', $event)"
              @update:layout-suggestion="emit('update:layout-suggestion', $event)"
            />
          </template>
        </ResizablePaneGroup>
      </template>

      <template #right>
        <ResizablePaneGroup
          storage-key="report-generation:right-column"
          class="h-full min-h-0"
          orientation="vertical"
          :pane-ids="['status', 'preview']"
          :default-sizes="[14, 86]"
          :min-sizes="[10, 32]"
        >
          <template #status>
            <ReportDraftStatusCard class="h-full" :saved-at-label="formattedSavedAt" :annotation-count="annotationTags.length" />
          </template>

          <template #preview>
            <ReportPreviewPanel
              class="h-full min-h-0"
              :patient="context.patient"
              :saved-at-label="formattedSavedAt"
              :findings="findings"
              :conclusion="conclusion"
              :layout-suggestion="layoutSuggestion"
              :annotation-count="annotationTags.length"
            />
          </template>
        </ResizablePaneGroup>
      </template>
    </ResizablePaneGroup>
  </section>
</template>