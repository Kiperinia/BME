<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'

import { getReportBuilderContext, preloadSam3Model, segmentFrameWithSam3 } from '@/api/reportBuilder'
import ReportBuilderWorkspace from '@/components/report/ReportBuilderWorkspace.vue'
import type { CaptureFramePayload, PolygonMask, ReportContextData, SegmentFrameResponse, TumorMaskData } from '@/types/eis'

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
 *   Handle player is playing.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const playerIsPlaying = ref(false)
/**
 * brief:
 *   Handle show mask.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const showMask = ref(true)
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
 *   Handle sam3 prompt text.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const sam3PromptText = ref('请优先关注镜头中央区域的可疑隆起病灶边界。')
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
  showMask.value = nextContext.showMask
  captureImages.value = [...nextContext.captureImageSrcs]
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
       *   - points: Input value for points.
       *   - frameWidth: Input value for frameWidth.
       *   - frameHeight: Input value for frameHeight.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const remoteContext = await getReportBuilderContext(props.reportId)
      applyContext(remoteContext)
    }
  } catch {
    showToast('报告构建上下文加载失败。', 'error')
  } finally {
    isHydrating.value = false
  }
}

/**
 * brief:
 *   Create sam3 mask.
 *
 * parameter:
 *   - points: Input value for points.
 *   - frameWidth: Input value for frameWidth.
 *   - frameHeight: Input value for frameHeight.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const createSam3Mask = (points: PolygonMask['points'], frameWidth: number, frameHeight: number): PolygonMask | null => {
  if (points.length < 3) {
    return null
  }

  return {
    id: `sam3-mask-${Date.now()}`,
    points,
    frameWidth,
    frameHeight,
    fillColor: 'rgba(16, 185, 129, 0.26)',
    strokeColor: 'rgba(5, 150, 105, 0.95)',
  }
}

/**
 * brief:
 *   Resolve sam3 mask data.
 *
 * parameter:
 *   - segmentation: Input value for segmentation.
 *   - frameWidth: Input value for frameWidth.
 *   - frameHeight: Input value for frameHeight.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const resolveSam3MaskData = (
  segmentation: SegmentFrameResponse,
  frameWidth: number,
  frameHeight: number,
): TumorMaskData => {
  if (segmentation.maskDataUrl) {
    return segmentation.maskDataUrl
  }

  /**
   * brief:
   *   Handle fallback mask.
   *
   * parameter:
   *   - payload: Input value for payload.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const fallbackMask = createSam3Mask(segmentation.maskCoordinates, frameWidth, frameHeight)
  return fallbackMask ? [fallbackMask] : ''
}

/**
 * brief:
 *   Handle capture frame.
 *
 * parameter:
 *   - payload: Input value for payload.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const handleCaptureFrame = async (payload: CaptureFramePayload) => {
  captureImages.value = [payload.dataUrl, ...captureImages.value].slice(0, 6)

  if (context.value) {
    context.value = {
      ...context.value,
      captureImageSrcs: [...captureImages.value],
    }
  }

  try {
    /**
     * brief:
     *   Handle segmentation.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const segmentation = await segmentFrameWithSam3(payload.dataUrl)
    /**
     * brief:
     *   Handle frame width.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const frameWidth = context.value?.videoFrameData.width ?? 1024
    /**
     * brief:
     *   Handle frame height.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const frameHeight = context.value?.videoFrameData.height ?? 1024
    /**
     * brief:
     *   Handle next mask data.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const nextMaskData = resolveSam3MaskData(segmentation, frameWidth, frameHeight)
    /**
     * brief:
     *   Handle has mask data.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const hasMaskData = Array.isArray(nextMaskData) ? nextMaskData.length > 0 : Boolean(nextMaskData)

    if (context.value && hasMaskData) {
      context.value = {
        ...context.value,
        showMask: true,
        maskData: nextMaskData,
        captureImageSrcs: [...captureImages.value],
        videoFrameData: {
          ...context.value.videoFrameData,
          timestamp: payload.capturedAt,
        },
        tumorFocus: {
          ...context.value.tumorFocus,
          tumorImageSrc: payload.dataUrl,
          maskData: nextMaskData,
          details: {
            ...context.value.tumorFocus.details,
            classification: 'SAM3 分割病灶候选',
            location: context.value.videoFrameData.suspectedLocation,
            confidence: 0.9,
          },
        },
      }
      showMask.value = true
      showToast('已抓取帧并完成 SAM3 分割。', 'success')
      return
    }
  } catch {
    showToast('已抓取帧，SAM3 分割暂不可用。', 'info')
    return
  }

  showToast(payload.includesMask ? '已抓取带遮罩帧。' : '已抓取原始帧。', 'success')
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
  preloadSam3Model().catch(() => undefined)
  await hydrateContext()
})
</script>

<template>
  <main class="mx-auto flex h-[calc(100vh-88px)] w-full max-w-[1920px] flex-col overflow-hidden px-4 py-3 lg:px-6 lg:py-4">
    <ReportBuilderWorkspace
      :context="context"
      :player-is-playing="playerIsPlaying"
      :show-mask="showMask"
      :capture-images="captureImages"
      :sam3-prompt-text="sam3PromptText"
      :is-hydrating="isHydrating"
      :toast-visible="toastVisible"
      :toast-message="toastMessage"
      :toast-tone="toastTone"
      @capture-frame="handleCaptureFrame"
      @patient-edit="handlePatientEdit"
      @view-history="handleViewHistory"
      @update:player-is-playing="playerIsPlaying = $event"
      @update:show-mask="showMask = $event"
      @update:sam3-prompt-text="sam3PromptText = $event"
    />
  </main>
</template>
