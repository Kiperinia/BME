<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'

import FeedbackToast from '@/components/common/FeedbackToast.vue'
import SegmentationViewer from '@/components/segmentation/SegmentationViewer.vue'
import ExemplarBankPanel from '@/components/workspace/ExemplarBankPanel.vue'
import ExemplarRetrievalPanel from '@/components/workspace/ExemplarRetrievalPanel.vue'
import ExpertConfigPanel from '@/components/workspace/ExpertConfigPanel.vue'
import PatientInfoPanel from '@/components/workspace/PatientInfoPanel.vue'
import ReportAgentPanel from '@/components/workspace/ReportAgentPanel.vue'
import { preloadWorkspaceSam3Model } from '@/api/workspace'
import { useWorkspaceStore } from '@/stores/workspace'

const workspaceStore = useWorkspaceStore()
const {
  patient,
  expertConfig,
  exemplarDecision,
  exemplarFeedback,
  exemplarRetrieval,
  isGeneratingReport,
  isEvaluatingExemplar,
  isRetrievingExemplars,
  isSegmenting,
  feedbackSubmittingFor,
  reportResult,
  segmentation,
  showMask,
  toast,
  uploadedImage,
  uploadedMaskFile,
  canEvaluateExemplar,
  canGenerateReport,
} = storeToRefs(workspaceStore)

const selectedFileSizeLabel = computed(() => {
  if (!uploadedImage.value) {
    return ''
  }

  return `${(uploadedImage.value.sizeBytes / 1024 / 1024).toFixed(2)} MB`
})

onMounted(() => {
  preloadWorkspaceSam3Model().catch(() => undefined)
})
</script>

<template>
  <main class="mx-auto flex min-h-[calc(100vh-88px)] w-full max-w-[1920px] flex-col gap-4 px-4 py-4 lg:px-6">
    <FeedbackToast :visible="toast.visible" :message="toast.message" :tone="toast.tone" />

    <PatientInfoPanel :model-value="patient" @update:model-value="workspaceStore.updatePatient" />

    <section class="grid items-start gap-4 2xl:grid-cols-[minmax(0,1.3fr)_520px]">
      <div class="grid content-start gap-4">
        <div class="w-full max-w-[1120px]">
          <SegmentationViewer
            :is-segmenting="isSegmenting"
            :has-mask-image="Boolean(uploadedMaskFile)"
            :image-url="uploadedImage?.objectUrl"
            :image-width="uploadedImage?.width"
            :image-height="uploadedImage?.height"
            :mask-data-url="segmentation?.maskDataUrl ?? ''"
            :mask-coordinates="segmentation?.maskCoordinates ?? []"
            :show-mask="showMask"
            :selected-filename="uploadedImage?.filename"
            :selected-file-size-label="selectedFileSizeLabel"
            :selected-mask-filename="uploadedMaskFile?.name"
            @select-file="workspaceStore.ingestLocalImage"
            @select-mask-file="workspaceStore.ingestMaskImage"
            @segment="workspaceStore.runSegmentation"
            @apply-mask="workspaceStore.applyUploadedMask"
            @toggle-mask="workspaceStore.toggleMask"
          />
        </div>

        <ExemplarRetrievalPanel
          :retrieval="exemplarRetrieval"
          :feedback-map="exemplarFeedback"
          :is-retrieving="isRetrievingExemplars"
          :feedback-submitting-for="feedbackSubmittingFor"
          @refresh="workspaceStore.refreshExemplarRetrieval"
          @feedback="workspaceStore.submitExemplarFeedback"
        />

        <ReportAgentPanel
          :report-result="reportResult"
          :is-generating="isGeneratingReport"
          :can-generate="canGenerateReport"
          @generate="workspaceStore.generateReport"
        />

        <ExemplarBankPanel
          :decision="exemplarDecision"
          :is-evaluating="isEvaluatingExemplar"
          :can-evaluate="canEvaluateExemplar"
          @evaluate="workspaceStore.evaluateExemplar"
        />
      </div>

      <aside class="grid content-start gap-4 2xl:sticky 2xl:top-4">
        <ExpertConfigPanel :model-value="expertConfig" @update:model-value="workspaceStore.updateExpertConfig" />
      </aside>
    </section>
  </main>
</template>
