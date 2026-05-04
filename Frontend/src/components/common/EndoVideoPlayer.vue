<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

import type { CaptureFramePayload, PolygonMask } from '@/types/eis'

type PlayStateSource = 'control' | 'video' | 'upload'

const uploadPromptBaseWidth = 520
const uploadPromptBaseHeight = 420

const props = withDefaults(
  defineProps<{
    videoSrc?: string
    isPlaying?: boolean
    maskData?: PolygonMask[]
    showMask?: boolean
  }>(),
  {
    videoSrc: '',
    isPlaying: false,
    maskData: () => [],
    showMask: true,
  },
)

const emit = defineEmits<{
  (event: 'play-state-change', payload: { isPlaying: boolean; source: PlayStateSource }): void
  (event: 'capture-frame', payload: CaptureFramePayload): void
  (event: 'update:showMask', showMask: boolean): void
}>()

const videoElement = ref<HTMLVideoElement>()
const canvasElement = ref<HTMLCanvasElement>()
const fileInputElement = ref<HTMLInputElement>()
const uploadViewportElement = ref<HTMLElement>()
const uploadedVideoUrl = ref('')
const hasMetadata = ref(false)
const isBuffering = ref(false)
const isDragging = ref(false)
const maskVisible = ref(props.showMask)
const mediaAspectRatio = ref(1)
const uploadPromptScale = ref(1)

let animationFrameId: number | null = null
let uploadResizeObserver: ResizeObserver | null = null

const currentVideoSrc = computed(() => props.videoSrc || uploadedVideoUrl.value)
const showUploadState = computed(() => !currentVideoSrc.value)
const uploadSurfaceStyle = computed(() => {
  return {
    width: `${uploadPromptBaseWidth * uploadPromptScale.value}px`,
    height: `${uploadPromptBaseHeight * uploadPromptScale.value}px`,
  }
})
const playbackLabel = computed(() => {
  if (!currentVideoSrc.value) {
    return '待上传'
  }

  return props.isPlaying ? '播放中' : '已暂停'
})

const revokeUploadedVideo = () => {
  if (uploadedVideoUrl.value) {
    URL.revokeObjectURL(uploadedVideoUrl.value)
    uploadedVideoUrl.value = ''
  }
}

const stopObservingUploadSurface = () => {
  if (uploadResizeObserver) {
    uploadResizeObserver.disconnect()
    uploadResizeObserver = null
  }
}

const syncUploadPromptScale = () => {
  const viewport = uploadViewportElement.value
  if (!viewport) {
    uploadPromptScale.value = 1
    return
  }

  const { width, height } = viewport.getBoundingClientRect()
  const nextScale = Math.min(width / uploadPromptBaseWidth, height / uploadPromptBaseHeight)
  uploadPromptScale.value = Math.max(0.76, Math.min(nextScale, 1.24))
}

const startObservingUploadSurface = async () => {
  stopObservingUploadSurface()

  await nextTick()
  syncUploadPromptScale()

  if (!uploadViewportElement.value || typeof ResizeObserver === 'undefined') {
    return
  }

  uploadResizeObserver = new ResizeObserver(() => {
    syncUploadPromptScale()
  })
  uploadResizeObserver.observe(uploadViewportElement.value)
}

const syncCanvasSize = () => {
  if (!canvasElement.value || !videoElement.value) {
    return
  }

  const canvas = canvasElement.value
  const video = videoElement.value
  const rect = video.getBoundingClientRect()
  const pixelRatio = window.devicePixelRatio || 1

  canvas.width = rect.width * pixelRatio
  canvas.height = rect.height * pixelRatio
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
}

const clearCanvas = () => {
  const canvas = canvasElement.value
  const context = canvas?.getContext('2d')

  if (!canvas || !context) {
    return
  }

  context.clearRect(0, 0, canvas.width, canvas.height)
}

const drawMask = () => {
  if (!canvasElement.value || !videoElement.value) {
    return
  }

  syncCanvasSize()
  clearCanvas()

  if (!maskVisible.value || !props.maskData.length || !hasMetadata.value) {
    return
  }

  const canvas = canvasElement.value
  const context = canvas.getContext('2d')

  if (!context) {
    return
  }

  const displayWidth = canvas.clientWidth
  const displayHeight = canvas.clientHeight

  for (const polygon of props.maskData) {
    if (!polygon.points.length) {
      continue
    }

    const scaleX = displayWidth / polygon.frameWidth
    const scaleY = displayHeight / polygon.frameHeight

    context.beginPath()
    polygon.points.forEach(([x, y], index) => {
      const targetX = x * scaleX
      const targetY = y * scaleY

      if (index === 0) {
        context.moveTo(targetX, targetY)
      } else {
        context.lineTo(targetX, targetY)
      }
    })
    context.closePath()
    context.fillStyle = polygon.fillColor ?? 'rgba(37, 99, 235, 0.26)'
    context.strokeStyle = polygon.strokeColor ?? 'rgba(37, 99, 235, 0.9)'
    context.lineWidth = 3
    context.fill()
    context.stroke()
  }
}

const stopDrawLoop = () => {
  if (animationFrameId !== null) {
    window.cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
}

const startDrawLoop = () => {
  stopDrawLoop()

  const step = () => {
    drawMask()

    if (videoElement.value && !videoElement.value.paused && !videoElement.value.ended) {
      animationFrameId = window.requestAnimationFrame(step)
    }
  }

  animationFrameId = window.requestAnimationFrame(step)
}

const playVideo = async (source: PlayStateSource) => {
  if (!videoElement.value || !currentVideoSrc.value) {
    return
  }

  try {
    await videoElement.value.play()
    startDrawLoop()
    emit('play-state-change', { isPlaying: true, source })
  } catch {
    emit('play-state-change', { isPlaying: false, source })
  }
}

const pauseVideo = (source: PlayStateSource) => {
  if (!videoElement.value) {
    return
  }

  videoElement.value.pause()
  stopDrawLoop()
  drawMask()
  emit('play-state-change', { isPlaying: false, source })
}

const handleTogglePlayback = async () => {
  if (!currentVideoSrc.value || !videoElement.value) {
    return
  }

  if (videoElement.value.paused) {
    await playVideo('control')
    return
  }

  pauseVideo('control')
}

const handleVideoSelected = async (file?: File) => {
  if (!file) {
    return
  }

  revokeUploadedVideo()
  uploadedVideoUrl.value = URL.createObjectURL(file)
  hasMetadata.value = false
  mediaAspectRatio.value = 1
  emit('play-state-change', { isPlaying: false, source: 'upload' })

  await nextTick()
  drawMask()
}

const syncVideoMetadata = () => {
  mediaAspectRatio.value = 1
}

const handleFileChange = async (event: Event) => {
  const target = event.target as HTMLInputElement
  await handleVideoSelected(target.files?.[0])
  target.value = ''
}

const handleDrop = async (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = false
  await handleVideoSelected(event.dataTransfer?.files?.[0])
}

const toggleMaskVisibility = () => {
  maskVisible.value = !maskVisible.value
  emit('update:showMask', maskVisible.value)
  drawMask()
}

const handleCaptureFrame = () => {
  if (!videoElement.value || !hasMetadata.value) {
    return
  }

  const video = videoElement.value
  const exportCanvas = document.createElement('canvas')
  exportCanvas.width = video.videoWidth || 1280
  exportCanvas.height = video.videoHeight || 720

  const context = exportCanvas.getContext('2d')
  if (!context) {
    return
  }

  context.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height)

  if (maskVisible.value) {
    for (const polygon of props.maskData) {
      if (!polygon.points.length) {
        continue
      }

      const scaleX = exportCanvas.width / polygon.frameWidth
      const scaleY = exportCanvas.height / polygon.frameHeight

      context.beginPath()
      polygon.points.forEach(([x, y], index) => {
        const targetX = x * scaleX
        const targetY = y * scaleY

        if (index === 0) {
          context.moveTo(targetX, targetY)
        } else {
          context.lineTo(targetX, targetY)
        }
      })
      context.closePath()
      context.fillStyle = polygon.fillColor ?? 'rgba(37, 99, 235, 0.26)'
      context.strokeStyle = polygon.strokeColor ?? 'rgba(37, 99, 235, 0.9)'
      context.lineWidth = 6
      context.fill()
      context.stroke()
    }
  }

  emit('capture-frame', {
    dataUrl: exportCanvas.toDataURL('image/png'),
    includesMask: maskVisible.value,
    capturedAt: video.currentTime,
  })
}

watch(
  () => props.showMask,
  (nextValue) => {
    maskVisible.value = nextValue
    drawMask()
  },
)

watch(
  () => currentVideoSrc.value,
  async () => {
    stopDrawLoop()
    hasMetadata.value = false
    isBuffering.value = false
    mediaAspectRatio.value = 1

    await nextTick()
    drawMask()

    if (props.isPlaying) {
      await playVideo('video')
    }
  },
)

watch(
  () => showUploadState.value,
  async (nextValue) => {
    if (!nextValue) {
      stopObservingUploadSurface()
      return
    }

    await startObservingUploadSurface()
  },
  { immediate: true },
)

watch(
  () => props.isPlaying,
  async (nextValue) => {
    if (!currentVideoSrc.value || !videoElement.value) {
      return
    }

    if (nextValue) {
      await playVideo('video')
      return
    }

    pauseVideo('video')
  },
)

onBeforeUnmount(() => {
  stopDrawLoop()
  stopObservingUploadSurface()
  revokeUploadedVideo()
})
</script>

<template>
  <section class="surface-card flex h-full min-h-0 flex-col overflow-hidden p-4">
    <div class="flex flex-col gap-3 border-b border-gray-100 pb-3 dark:border-slate-700 md:flex-row md:items-center md:justify-between">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">内镜视频播放器</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          视频层与 Canvas 遮罩层保持像素对齐，用于承载 SAM3 分割结果。
        </p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-200">
        {{ playbackLabel }}
      </span>
    </div>

    <div class="mt-4 flex min-h-0 flex-1 items-start justify-center">
      <div v-if="showUploadState" ref="uploadViewportElement" class="flex h-full w-full items-center justify-center">
        <div
          class="relative flex items-center justify-center rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 px-3 text-center transition dark:border-slate-600 dark:bg-slate-900"
          :style="uploadSurfaceStyle"
          :class="isDragging ? 'border-blue-500 bg-blue-50 dark:border-sky-400 dark:bg-sky-950/40' : ''"
          @click="fileInputElement?.click()"
          @dragenter.prevent="isDragging = true"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop="handleDrop"
        >
          <input
            ref="fileInputElement"
            class="hidden"
            type="file"
            accept="video/*"
            @change="handleFileChange"
          />
          <div
            class="space-y-1"
            :style="{ transform: `scale(${uploadPromptScale})`, transformOrigin: 'center center' }"
          >
            <div class="mx-auto flex h-9 w-9 items-center justify-center rounded-full bg-white text-blue-600 shadow-sm dark:bg-slate-800 dark:text-sky-300">
              <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1.5" />
              </svg>
            </div>
            <div>
              <p class="text-sm font-medium text-gray-700 dark:text-gray-100">拖拽或点击上传内窥镜视频</p>
              <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
                支持本地历史视频回放，上传后即可叠加测试用分割多边形。
              </p>
            </div>
          </div>
        </div>
      </div>

      <div
        v-else
        class="relative aspect-square w-full max-w-[760px] overflow-hidden rounded-2xl bg-slate-950"
        :style="{ aspectRatio: `${mediaAspectRatio}` }"
      >
        <video
          ref="videoElement"
          class="h-full w-full object-cover"
          :src="currentVideoSrc"
          playsinline
          controlslist="nodownload"
          preload="metadata"
          @canplay="isBuffering = false"
          @ended="pauseVideo('video')"
          @loadedmetadata="hasMetadata = true; syncVideoMetadata(); syncCanvasSize(); drawMask()"
          @pause="pauseVideo('video')"
          @play="startDrawLoop(); emit('play-state-change', { isPlaying: true, source: 'video' })"
          @timeupdate="drawMask()"
          @waiting="isBuffering = true"
        />

        <canvas ref="canvasElement" class="pointer-events-none absolute inset-0 h-full w-full" />

        <div
          v-if="isBuffering"
          class="absolute inset-0 flex items-center justify-center bg-slate-950/50"
        >
          <div class="flex items-center gap-3 rounded-full bg-slate-900/80 px-4 py-2 text-sm text-white">
            <span class="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
            视频缓冲中
          </div>
        </div>
      </div>
    </div>

    <div class="mt-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
      <div class="flex flex-wrap gap-3">
        <button type="button" class="surface-button-primary px-3 py-2 text-sm" @click="handleTogglePlayback">
          {{ props.isPlaying ? '暂停' : '播放' }}
        </button>
        <button type="button" class="surface-button-secondary px-3 py-2 text-sm" @click="toggleMaskVisibility">
          {{ maskVisible ? '隐藏遮罩' : '显示遮罩' }}
        </button>
        <button type="button" class="surface-button-secondary px-3 py-2 text-sm" @click="handleCaptureFrame">
          抓帧
        </button>
      </div>

      <p class="text-xs text-gray-500 dark:text-gray-400 md:text-sm">
        当前测试遮罩点位：{{ props.maskData.length }} 组
      </p>
    </div>
  </section>
</template>