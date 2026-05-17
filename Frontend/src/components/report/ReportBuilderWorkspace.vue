<script setup lang="ts">
import EndoVideoPlayer from '@/components/common/EndoVideoPlayer.vue'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import PatientInfoCard from '@/components/common/PatientInfoCard.vue'
import ResizablePaneGroup from '@/components/common/ResizablePaneGroup.vue'
import ReportCaptureGallery from '@/components/report/ReportCaptureGallery.vue'
import Sam3PromptPanel from '@/components/report/Sam3PromptPanel.vue'
import type { CaptureFramePayload, ReportContextData } from '@/types/eis'

const props = defineProps<{
  context: ReportContextData | null
  playerIsPlaying: boolean
  showMask: boolean
  captureImages: string[]
  sam3PromptText: string
  isHydrating: boolean
  toastVisible: boolean
  toastMessage: string
  toastTone: 'info' | 'success' | 'error'
}>()

const emit = defineEmits<{
  (event: 'capture-frame', payload: CaptureFramePayload): void
  (event: 'patient-edit', patientId: string): void
  (event: 'view-history'): void
  (event: 'update:player-is-playing', value: boolean): void
  (event: 'update:show-mask', value: boolean): void
  (event: 'update:sam3-prompt-text', value: string): void
}>()
</script>

<template>
  <section class="relative flex h-full min-h-0 flex-col gap-3">
    <FeedbackToast :visible="toastVisible" :message="toastMessage" :tone="toastTone" />

    <div class="flex flex-none flex-col gap-1.5">
      <div>
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">检查工作台</h2>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
          当前页面只保留患者基本信息、内窥镜视频与抓帧切片，聚焦检查与采集流程。
        </p>
      </div>
    </div>

    <div v-if="isHydrating && !context" class="surface-card p-6">
      <div class="space-y-4 animate-pulse">
        <div class="h-6 w-48 rounded bg-gray-100 dark:bg-slate-700" />
        <div class="h-64 rounded-2xl bg-gray-100 dark:bg-slate-800" />
        <div class="h-48 rounded-2xl bg-gray-100 dark:bg-slate-800" />
      </div>
    </div>

    <ResizablePaneGroup
      v-else-if="context"
      storage-key="report-builder:main-layout"
      class="min-h-0 flex-1"
      orientation="horizontal"
      :pane-ids="['left', 'right']"
      :default-sizes="[24, 76]"
      :min-sizes="[16, 38]"
      :collapse-below="1280"
    >
      <template #left>
        <ResizablePaneGroup
          storage-key="report-builder:left-column"
          class="h-full min-h-0"
          orientation="vertical"
          :pane-ids="['patient', 'prompt']"
          :default-sizes="[42, 58]"
          :min-sizes="[24, 20]"
        >
          <template #patient>
            <PatientInfoCard
              class="aspect-square h-auto min-h-0 w-full overflow-hidden"
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

          <template #prompt>
            <Sam3PromptPanel
              class="h-full min-h-0"
              :polyp-count="Array.isArray(context.maskData) ? context.maskData.length : (context.maskData ? 1 : 0)"
              :prompt-text="sam3PromptText"
              @update:prompt-text="emit('update:sam3-prompt-text', $event)"
            />
          </template>
        </ResizablePaneGroup>
      </template>

      <template #right>
        <ResizablePaneGroup
          storage-key="report-builder:right-column"
          class="h-full min-h-0"
          orientation="vertical"
          :pane-ids="['video', 'capture']"
          :default-sizes="[62, 38]"
          :min-sizes="[32, 18]"
        >
          <template #video>
            <EndoVideoPlayer
              class="h-full min-h-0 w-full"
              :video-src="context.videoSrc"
              :is-playing="playerIsPlaying"
              :mask-data="context.maskData"
              :show-mask="showMask"
              @capture-frame="emit('capture-frame', $event)"
              @play-state-change="emit('update:player-is-playing', $event.isPlaying)"
              @update:show-mask="emit('update:show-mask', $event)"
            />
          </template>

          <template #capture>
            <ReportCaptureGallery class="h-full min-h-0 w-full" :images="captureImages" />
          </template>
        </ResizablePaneGroup>
      </template>
    </ResizablePaneGroup>
  </section>
</template>