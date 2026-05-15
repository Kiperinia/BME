<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'

import FeedbackToast from '@/components/common/FeedbackToast.vue'
import ImageUploadPanel from '@/components/segmentation/ImageUploadPanel.vue'
import MaskToggle from '@/components/segmentation/MaskToggle.vue'
import SegmentationInfoPanel from '@/components/segmentation/SegmentationInfoPanel.vue'
import SegmentationViewer from '@/components/segmentation/SegmentationViewer.vue'
import ExemplarBankPanel from '@/components/workspace/ExemplarBankPanel.vue'
import ExpertConfigPanel from '@/components/workspace/ExpertConfigPanel.vue'
import ReportAgentPanel from '@/components/workspace/ReportAgentPanel.vue'
import { useWorkspaceStore } from '@/stores/workspace'

const workspaceStore = useWorkspaceStore()
const {
  expertConfig,
  exemplarDecision,
  isEvaluatingExemplar,
  isGeneratingReport,
  isSegmenting,
  reportResult,
  segmentation,
  showMask,
  toast,
  uploadedImage,
  canEvaluateExemplar,
  canGenerateReport,
  canSegment,
} = storeToRefs(workspaceStore)

const selectedFileSizeLabel = computed(() => {
  if (!uploadedImage.value) {
    return ''
  }

  return `${(uploadedImage.value.sizeBytes / 1024 / 1024).toFixed(2)} MB`
})
</script>

<template>
  <main class="mx-auto flex min-h-[calc(100vh-88px)] w-full max-w-[1920px] flex-col gap-4 px-4 py-4 lg:px-6">
    <FeedbackToast :visible="toast.visible" :message="toast.message" :tone="toast.tone" />

    <ImageUploadPanel
      :has-image="Boolean(uploadedImage)"
      :is-segmenting="isSegmenting"
      :selected-filename="uploadedImage?.filename"
      :selected-file-size-label="selectedFileSizeLabel"
      @select-file="workspaceStore.ingestLocalImage"
      @segment="workspaceStore.runSegmentation"
    />

    <section class="grid gap-4 2xl:grid-cols-[minmax(0,1.6fr)_420px]">
      <div class="grid gap-4">
        <div class="flex items-center justify-between gap-3">
          <div>
            <p class="text-sm font-medium text-slate-700 dark:text-slate-200">结果视图</p>
            <p class="text-sm text-slate-500 dark:text-slate-400">上传完成后可在此查看原图、分割图和掩码叠加。</p>
          </div>
          <MaskToggle :enabled="showMask" :disabled="!segmentation" @toggle="workspaceStore.toggleMask" />
        </div>

        <SegmentationViewer
          :image-url="uploadedImage?.objectUrl"
          :image-width="uploadedImage?.width"
          :image-height="uploadedImage?.height"
          :mask-coordinates="segmentation?.maskCoordinates ?? []"
          :show-mask="showMask"
        />

        <ReportAgentPanel
          :report-result="reportResult"
          :is-generating="isGeneratingReport"
          :can-generate="canGenerateReport && canSegment"
          @generate="workspaceStore.generateReport"
        />

        <ExemplarBankPanel
          :decision="exemplarDecision"
          :is-evaluating="isEvaluatingExemplar"
          :can-evaluate="canEvaluateExemplar && canSegment"
          @evaluate="workspaceStore.evaluateExemplar"
        />
      </div>

      <aside class="grid content-start gap-4">
        <SegmentationInfoPanel :image="uploadedImage" :segmentation="segmentation" :show-mask="showMask" />
        <ExpertConfigPanel :model-value="expertConfig" @update:model-value="workspaceStore.updateExpertConfig" />
      </aside>
    </section>
  </main>
</template>
