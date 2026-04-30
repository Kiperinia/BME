<script setup lang="ts">
import { computed } from 'vue'

import EndoVideoPlayer from '@/components/common/EndoVideoPlayer.vue'
import FeedbackToast from '@/components/common/FeedbackToast.vue'
import PatientInfoCard from '@/components/common/PatientInfoCard.vue'
import ReportCaptureGallery from '@/components/report/ReportCaptureGallery.vue'
import type { CaptureFramePayload, ReportContextData } from '@/types/eis'

const props = defineProps<{
  context: ReportContextData | null
  playerIsPlaying: boolean
  showMask: boolean
  captureImages: string[]
  videoAspectRatio: number
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
  (event: 'video-aspect-ratio-change', value: number): void
}>()

const rightPaneRowStyle = computed(() => {
  const aspectRatio = props.videoAspectRatio || 4 / 3

  if (aspectRatio >= 1.6) {
    return {
      gridTemplateRows: 'minmax(0, 1.4fr) minmax(0, 0.6fr)',
    }
  }

  if (aspectRatio >= 1.33) {
    return {
      gridTemplateRows: 'minmax(0, 1.25fr) minmax(0, 0.75fr)',
    }
  }

  return {
    gridTemplateRows: 'minmax(0, 1.1fr) minmax(0, 0.9fr)',
  }
})
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

    <div v-else-if="context" class="grid min-h-0 flex-1 gap-3 xl:grid-cols-[300px_minmax(0,1fr)]">
      <PatientInfoCard
        class="h-full min-h-0 w-full overflow-hidden"
        :patient-name="context.patient.patientName"
        :gender="context.patient.gender"
        :age="context.patient.age"
        :patient-id="context.patient.patientId"
        :exam-date="context.patient.examDate"
        :status="context.patient.status"
        @edit="emit('patient-edit', $event)"
        @view-history="emit('view-history')"
      />

      <div class="grid min-h-0 h-full w-full gap-3" :style="rightPaneRowStyle">
        <EndoVideoPlayer
          class="min-h-0 h-full"
          :video-src="context.videoSrc"
          :is-playing="playerIsPlaying"
          :mask-data="context.maskData"
          :show-mask="showMask"
          @capture-frame="emit('capture-frame', $event)"
          @play-state-change="emit('update:player-is-playing', $event.isPlaying)"
          @update:show-mask="emit('update:show-mask', $event)"
          @video-metadata-change="emit('video-aspect-ratio-change', $event.aspectRatio)"
        />

        <ReportCaptureGallery class="min-h-0 w-full" :images="captureImages" />
      </div>
    </div>
  </section>
</template>