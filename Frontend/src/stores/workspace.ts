import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

import {
  evaluateExemplarCandidate,
  generateWorkspaceReport,
  retrieveExemplarPrior,
  sendExemplarFeedback,
  segmentWorkspaceImage,
} from '@/api/workspace'
import { usePatientRecordsStore } from '@/stores/patientRecords'
import {
  createFormalPatientId,
  createDefaultExpertConfiguration,
  createDefaultPatient,
  formatParisClassification,
  type ExemplarBankDecision,
  type ExemplarFeedbackMode,
  type ExemplarFeedbackResult,
  type ExemplarRetrievalResult,
  type ExpertConfiguration,
  type ToastState,
  type UploadedWorkspaceImage,
  type WorkspacePatient,
  type WorkspaceReportRequest,
  type WorkspaceReportResult,
  type WorkspaceSegmentation,
} from '@/types/workspace'

/**
 * brief:
 *   Read file as data url.
 *
 * parameter:
 *   - file: Input value for file.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const readFileAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    /**
     * brief:
     *   Handle reader.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(new Error('failed to read local image'))
    reader.readAsDataURL(file)
  })

/**
 * brief:
 *   Measure image.
 *
 * parameter:
 *   - src: Input value for src.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const measureImage = (src: string) =>
  new Promise<{ width: number; height: number }>((resolve, reject) => {
    /**
     * brief:
     *   Handle image.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const image = new Image()
    image.onload = () => {
      resolve({
        width: image.naturalWidth || 1,
        height: image.naturalHeight || 1,
      })
    }
    image.onerror = () => reject(new Error('failed to decode image dimensions'))
    image.src = src
  })

/**
 * brief:
 *   Resolve request error message.
 *
 * parameter:
 *   - error: Input value for error.
 *   - fallbackMessage: Input value for fallbackMessage.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const resolveRequestErrorMessage = (error: unknown, fallbackMessage: string) => {
  if (axios.isAxiosError(error)) {
    /**
     * brief:
     *   Handle api message.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const apiMessage = error.response?.data?.message
    if (typeof apiMessage === 'string' && apiMessage.trim()) {
      return apiMessage
    }

    /**
     * brief:
     *   Handle detail message.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const detailMessage = error.response?.data?.detail?.message
    if (typeof detailMessage === 'string' && detailMessage.trim()) {
      return detailMessage
    }

    if (typeof error.message === 'string' && error.message.trim()) {
      return error.message
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message
  }

  return fallbackMessage
}

/**
 * brief:
 *   Handle use workspace store.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const useWorkspaceStore = defineStore('workspace', () => {
  /**
   * brief:
   *   Handle patient records store.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const patientRecordsStore = usePatientRecordsStore()
  /**
   * brief:
   *   Handle patient.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const patient = ref(createDefaultPatient())
  /**
   * brief:
   *   Handle expert config.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const expertConfig = ref(createDefaultExpertConfiguration())
  /**
   * brief:
   *   Handle uploaded file.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const uploadedFile = ref<File | null>(null)
  /**
   * brief:
   *   Handle uploaded mask file.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const uploadedMaskFile = ref<File | null>(null)
  /**
   * brief:
   *   Handle uploaded image.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const uploadedImage = ref<UploadedWorkspaceImage | null>(null)
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
  const segmentation = ref<WorkspaceSegmentation | null>(null)
  /**
   * brief:
   *   Handle report result.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const reportResult = ref<WorkspaceReportResult | null>(null)
  /**
   * brief:
   *   Handle exemplar decision.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const exemplarDecision = ref<ExemplarBankDecision | null>(null)
  /**
   * brief:
   *   Handle exemplar retrieval.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const exemplarRetrieval = ref<ExemplarRetrievalResult | null>(null)
  /**
   * brief:
   *   Handle exemplar feedback.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const exemplarFeedback = ref<Record<string, ExemplarFeedbackResult>>({})
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
   *   Handle is segmenting.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isSegmenting = ref(false)
  /**
   * brief:
   *   Handle is generating report.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isGeneratingReport = ref(false)
  /**
   * brief:
   *   Handle is evaluating exemplar.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isEvaluatingExemplar = ref(false)
  /**
   * brief:
   *   Handle is retrieving exemplars.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isRetrievingExemplars = ref(false)
  /**
   * brief:
   *   Handle feedback submitting for.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const feedbackSubmittingFor = ref<string | null>(null)
  /**
   * brief:
   *   Handle toast.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const toast = ref<ToastState>({
    visible: false,
    message: '',
    tone: 'info',
  })

  let toastTimer: number | undefined

  /**
   * brief:
   *   Handle can segment.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const canSegment = computed(() => Boolean(uploadedFile.value && uploadedImage.value))
  /**
   * brief:
   *   Handle can generate report.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const canGenerateReport = computed(() => Boolean(uploadedImage.value))
  /**
   * brief:
   *   Handle can evaluate exemplar.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const canEvaluateExemplar = computed(() => Boolean(uploadedImage.value && (segmentation.value?.maskDataUrl || segmentation.value?.maskCoordinates.length)))

  /**
   * brief:
   *   Handle push toast.
   *
   * parameter:
   *   - message: Input value for message.
   *   - tone: Input value for tone.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const pushToast = (message: string, tone: ToastState['tone'] = 'info') => {
    toast.value = {
      visible: true,
      message,
      tone,
    }

    if (toastTimer) {
      window.clearTimeout(toastTimer)
    }

    toastTimer = window.setTimeout(() => {
      toast.value.visible = false
    }, 2600)
  }

  /**
   * brief:
   *   Reset workflow state.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const resetWorkflowState = () => {
    segmentation.value = null
    reportResult.value = null
    exemplarDecision.value = null
    exemplarRetrieval.value = null
    exemplarFeedback.value = {}
    showMask.value = true
  }

  /**
   * brief:
   *   Update patient.
   *
   * parameter:
   *   - nextValue: Input value for nextValue.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const updatePatient = (nextValue: WorkspacePatient) => {
    patient.value = {
      ...nextValue,
      patientId: nextValue.patientId.trim() || patient.value.patientId,
    }
  }

  /**
   * brief:
   *   Update expert config.
   *
   * parameter:
   *   - nextValue: Input value for nextValue.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const updateExpertConfig = (nextValue: ExpertConfiguration) => {
    expertConfig.value = {
      ...nextValue,
      parisClassification: formatParisClassification(nextValue.parisDetail),
    }
  }

  /**
   * brief:
   *   Handle revoke current object url.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const revokeCurrentObjectUrl = () => {
    if (uploadedImage.value?.objectUrl) {
      URL.revokeObjectURL(uploadedImage.value.objectUrl)
    }
  }

  /**
   * brief:
   *   Ingest local image.
   *
   * parameter:
   *   - file: Input value for file.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const ingestLocalImage = async (file: File) => {
    revokeCurrentObjectUrl()

    /**
     * brief:
     *   Handle object url.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const objectUrl = URL.createObjectURL(file)
    const [dataUrl, dimensions] = await Promise.all([readFileAsDataUrl(file), measureImage(objectUrl)])
    uploadedFile.value = file
    uploadedImage.value = {
      filename: file.name,
      contentType: file.type || 'image/png',
      dataUrl,
      objectUrl,
      width: dimensions.width,
      height: dimensions.height,
      sizeBytes: file.size,
    }

    patient.value = {
      ...patient.value,
      patientId: !patient.value.patientId || /^case-/i.test(patient.value.patientId)
        ? createFormalPatientId()
        : patient.value.patientId,
    }
    resetWorkflowState()
    uploadedMaskFile.value = null
    pushToast('本地图像已载入，可以开始 MedicalSAM3 分割。', 'success')
  }

  /**
   * brief:
   *   Ingest mask image.
   *
   * parameter:
   *   - file: Input value for file.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const ingestMaskImage = (file: File) => {
    uploadedMaskFile.value = file
    pushToast('掩码图已载入，可点击“应用掩码展示”。', 'success')
  }

  /**
   * brief:
   *   Handle parse mask data from image.
   *
   * parameter:
   *   - imageSource: Input value for imageSource.
   *   - dimensions: Input value for dimensions.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const parseMaskDataFromImage = async (
    imageSource: string,
    dimensions: { width: number; height: number },
  ): Promise<WorkspaceSegmentation> => {
    /**
     * brief:
     *   Handle image.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      /**
       * brief:
       *   Handle instance.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const instance = new Image()
      instance.onload = () => resolve(instance)
      instance.onerror = () => reject(new Error('failed to decode uploaded mask image'))
      instance.src = imageSource
    })

    /**
     * brief:
     *   Handle canvas.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const canvas = document.createElement('canvas')
    canvas.width = dimensions.width
    canvas.height = dimensions.height
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
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('failed to build mask parser canvas context')
    }

    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    /**
     * brief:
     *   Handle image data.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
    /**
     * brief:
     *   Handle data.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const data = imageData.data
    /**
     * brief:
     *   Handle overlay.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const overlay = context.createImageData(canvas.width, canvas.height)
    /**
     * brief:
     *   Handle overlay data.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const overlayData = overlay.data

    let borderLuminanceSum = 0
    let borderPixelCount = 0
    let transparentPixelCount = 0
    let brightPixelCount = 0
    let darkPixelCount = 0

    for (let index = 0; index < data.length; index += 4) {
      /**
       * brief:
       *   Handle pixel index.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const pixelIndex = index / 4
      /**
       * brief:
       *   Handle x.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const x = pixelIndex % canvas.width
      /**
       * brief:
       *   Handle y.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const y = Math.floor(pixelIndex / canvas.width)
      /**
       * brief:
       *   Handle r.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const r = data[index] ?? 0
      /**
       * brief:
       *   Handle g.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const g = data[index + 1] ?? 0
      /**
       * brief:
       *   Handle b.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const b = data[index + 2] ?? 0
      /**
       * brief:
       *   Handle alpha.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const alpha = data[index + 3] ?? 0
      /**
       * brief:
       *   Handle luminance.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

      if (x === 0 || y === 0 || x === canvas.width - 1 || y === canvas.height - 1) {
        borderLuminanceSum += luminance
        borderPixelCount += 1
      }

      if (alpha < 240) {
        transparentPixelCount += 1
      }

      if (alpha > 20) {
        if (luminance >= 128) {
          brightPixelCount += 1
        } else {
          darkPixelCount += 1
        }
      }
    }

    /**
     * brief:
     *   Handle border mean luminance.
     *
     * parameter:
     *   - predicate: Input value for predicate.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const borderMeanLuminance = borderPixelCount > 0 ? borderLuminanceSum / borderPixelCount : 0
    /**
     * brief:
     *   Handle background is bright.
     *
     * parameter:
     *   - predicate: Input value for predicate.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const backgroundIsBright = borderMeanLuminance >= 128
    /**
     * brief:
     *   Handle has transparent mask.
     *
     * parameter:
     *   - predicate: Input value for predicate.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const hasTransparentMask = transparentPixelCount > data.length / 4 * 0.01
    /**
     * brief:
     *   Handle minority is bright.
     *
     * parameter:
     *   - predicate: Input value for predicate.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const minorityIsBright = brightPixelCount > 0 && brightPixelCount <= darkPixelCount

    /**
     * brief:
     *   Handle paint overlay.
     *
     * parameter:
     *   - predicate: Input value for predicate.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const paintOverlay = (predicate: (luminance: number, alpha: number) => boolean) => {
      let minX = canvas.width
      let minY = canvas.height
      let maxX = 0
      let maxY = 0
      let foregroundCount = 0

      for (let index = 0; index < data.length; index += 4) {
        /**
         * brief:
         *   Handle pixel index.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const pixelIndex = index / 4
        /**
         * brief:
         *   Handle x.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const x = pixelIndex % canvas.width
        /**
         * brief:
         *   Handle y.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const y = Math.floor(pixelIndex / canvas.width)
        /**
         * brief:
         *   Handle r.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const r = data[index] ?? 0
        /**
         * brief:
         *   Handle g.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const g = data[index + 1] ?? 0
        /**
         * brief:
         *   Handle b.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const b = data[index + 2] ?? 0
        /**
         * brief:
         *   Handle alpha.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const alpha = data[index + 3] ?? 0
        /**
         * brief:
         *   Handle luminance.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        /**
         * brief:
         *   Handle is foreground.
         *
         * parameter:
         *   - None.
         *
         * retrival:
         *   - Returns the computed value or updates local application state.
         */
        const isForeground = predicate(luminance, alpha)

        if (!isForeground) {
          overlayData[index] = 0
          overlayData[index + 1] = 0
          overlayData[index + 2] = 0
          overlayData[index + 3] = 0
          continue
        }

        overlayData[index] = 56
        overlayData[index + 1] = 189
        overlayData[index + 2] = 248
        overlayData[index + 3] = 172
        foregroundCount += 1
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
      }

      return { minX, minY, maxX, maxY, foregroundCount }
    }

    /**
     * brief:
     *   Handle primary result.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const primaryResult = paintOverlay((luminance, alpha) => {
      if (hasTransparentMask) {
        return alpha > 20
      }
      return backgroundIsBright ? luminance < 200 : luminance > 55
    })

    /**
     * brief:
     *   Handle pixel count.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const pixelCount = Math.max(canvas.width * canvas.height, 1)
    /**
     * brief:
     *   Handle primary ratio.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const primaryRatio = primaryResult.foregroundCount / pixelCount
    /**
     * brief:
     *   Handle needs fallback threshold.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const needsFallbackThreshold = primaryRatio <= 0.001 || primaryRatio >= 0.98

    const { minX, minY, maxX, maxY, foregroundCount } = needsFallbackThreshold
      ? paintOverlay((luminance, alpha) => {
          if (alpha <= 20) {
            return false
          }
          return minorityIsBright ? luminance >= 128 : luminance < 128
        })
      : primaryResult

    if (foregroundCount === 0) {
      throw new Error('uploaded mask image has no detectable foreground area')
    }

    /**
     * brief:
     *   Handle overlay canvas.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const overlayCanvas = document.createElement('canvas')
    overlayCanvas.width = canvas.width
    overlayCanvas.height = canvas.height
    /**
     * brief:
     *   Handle overlay context.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const overlayContext = overlayCanvas.getContext('2d')
    if (!overlayContext) {
      throw new Error('failed to build mask overlay canvas context')
    }
    overlayContext.putImageData(overlay, 0, 0)

    /**
     * brief:
     *   Handle image area.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const imageArea = Math.max(canvas.width * canvas.height, 1)
    const maskCoordinates: [number, number][] = [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY],
    ]
    return {
      maskDataUrl: overlayCanvas.toDataURL('image/png'),
      maskCoordinates,
      boundingBox: [minX, minY, maxX, maxY],
      maskAreaPixels: foregroundCount,
      maskAreaRatio: foregroundCount / imageArea,
      pointCount: maskCoordinates.length,
    }
  }

  /**
   * brief:
   *   Apply uploaded mask.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const applyUploadedMask = async () => {
    if (!uploadedMaskFile.value || !uploadedImage.value) {
      pushToast('请先上传原图和掩码图。', 'error')
      return
    }

    try {
      /**
       * brief:
       *   Handle mask data url.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const maskDataUrl = await readFileAsDataUrl(uploadedMaskFile.value)
      segmentation.value = await parseMaskDataFromImage(maskDataUrl, {
        width: uploadedImage.value.width,
        height: uploadedImage.value.height,
      })
      showMask.value = true
      reportResult.value = null
      exemplarDecision.value = null
      await refreshExemplarRetrieval()
      pushToast('已应用上传掩码，可直接生成报告。', 'success')
    } catch (error) {
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, '掩码图解析失败，请更换文件后重试。')
      pushToast(message, 'error')
    }
  }

  /**
   * brief:
   *   Handle run segmentation.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const runSegmentation = async () => {
    if (!uploadedFile.value || !uploadedImage.value) {
      pushToast('请先选择一张本地图像。', 'error')
      return
    }

    isSegmenting.value = true
    try {
      /**
       * brief:
       *   Handle result.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const result = await segmentWorkspaceImage(uploadedFile.value, {
        width: uploadedImage.value.width,
        height: uploadedImage.value.height,
      }, {
        patient: patient.value,
        expertConfig: {
          ...expertConfig.value,
          parisClassification: formatParisClassification(expertConfig.value.parisDetail),
        },
        bankId: 'default-bank',
        topK: 6,
      })
      segmentation.value = result
      showMask.value = true
      reportResult.value = null
      exemplarDecision.value = null
      await refreshExemplarRetrieval()
      pushToast('MedicalSAM3 分割完成。', 'success')
    } catch (error) {
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, '分割失败，请检查后端服务。')
      pushToast(message, 'error')
    } finally {
      isSegmenting.value = false
    }
  }

  /**
   * brief:
   *   Build report request.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const buildReportRequest = (): WorkspaceReportRequest | null => {
    if (!uploadedImage.value) {
      return null
    }

    /**
     * brief:
     *   Handle resolved segmentation.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const resolvedSegmentation = segmentation.value ?? {
      maskDataUrl: '',
      maskCoordinates: [] as [number, number][],
      boundingBox: [0, 0, 0, 0] as [number, number, number, number],
      maskAreaPixels: 0,
      maskAreaRatio: 0,
      pointCount: 0,
    }

    return {
      patient: patient.value,
      image: {
        filename: uploadedImage.value.filename,
        contentType: uploadedImage.value.contentType,
        dataUrl: uploadedImage.value.dataUrl,
        width: uploadedImage.value.width,
        height: uploadedImage.value.height,
      },
      segmentation: resolvedSegmentation,
      expertConfig: {
        ...expertConfig.value,
        parisClassification: formatParisClassification(expertConfig.value.parisDetail),
      },
    }
  }

  /**
   * brief:
   *   Handle refresh exemplar retrieval.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const refreshExemplarRetrieval = async () => {
    /**
     * brief:
     *   Handle payload.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const payload = buildReportRequest()
    if (!payload) {
      exemplarRetrieval.value = null
      return
    }

    isRetrievingExemplars.value = true
    try {
      exemplarRetrieval.value = await retrieveExemplarPrior({
        ...payload,
        topK: 6,
        bankId: segmentation.value?.retrievalBankId ?? 'default-bank',
      })
    } catch (error) {
      exemplarRetrieval.value = null
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, 'Exemplar retrieval failed.')
      pushToast(message, 'error')
    } finally {
      isRetrievingExemplars.value = false
    }
  }

  /**
   * brief:
   *   Generate report.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const generateReport = async () => {
    /**
     * brief:
     *   Handle payload.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const payload = buildReportRequest()
    if (!payload) {
      pushToast('请先完成图像上传和分割。', 'error')
      return
    }

    isGeneratingReport.value = true
    try {
      reportResult.value = await generateWorkspaceReport(payload)
      exemplarDecision.value = null
      patientRecordsStore.addRecord({
        patient: payload.patient,
        imageFilename: payload.image.filename,
        findings: reportResult.value.findings,
        conclusion: reportResult.value.conclusion,
        recommendation: reportResult.value.recommendation,
        reportMarkdown: reportResult.value.reportMarkdown,
        featureTags: reportResult.value.featureTags,
        parisClassification: payload.expertConfig.parisClassification,
        lesionType: payload.expertConfig.lesionType,
        pathologyClassification: payload.expertConfig.pathologyClassification,
        workflowMode: reportResult.value.workflow.workflowMode,
        riskLevel: reportResult.value.workflow.lesions[0]?.riskLevel ?? '',
      })
      pushToast('正式诊断报告已生成，并已写入病例索引。', 'success')
    } catch (error) {
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, '报告生成失败。')
      pushToast(message, 'error')
    } finally {
      isGeneratingReport.value = false
    }
  }

  /**
   * brief:
   *   Evaluate exemplar.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const evaluateExemplar = async () => {
    /**
     * brief:
     *   Handle payload.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const payload = buildReportRequest()
    if (!payload) {
      pushToast('请先完成图像上传和分割。', 'error')
      return
    }

    isEvaluatingExemplar.value = true
    try {
      exemplarDecision.value = await evaluateExemplarCandidate({
        ...payload,
        polarityHint: 'positive',
        reportMarkdown: reportResult.value?.reportMarkdown ?? '',
        findings: reportResult.value?.findings ?? '',
        conclusion: reportResult.value?.conclusion ?? '',
      })
      pushToast(
        exemplarDecision.value.accepted
          ? '样本已加入 exemplar bank。'
          : '样本已评估，当前未进入 exemplar bank。',
        exemplarDecision.value.accepted ? 'success' : 'info',
      )
    } catch (error) {
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - exemplarId: Input value for exemplarId.
       *   - failureMode: Input value for failureMode.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, '样本库评估失败。')
      pushToast(message, 'error')
    } finally {
      isEvaluatingExemplar.value = false
    }
  }

  /**
   * brief:
   *   Handle submit exemplar feedback.
   *
   * parameter:
   *   - exemplarId: Input value for exemplarId.
   *   - failureMode: Input value for failureMode.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const submitExemplarFeedback = async (exemplarId: string, failureMode: ExemplarFeedbackMode) => {
    /**
     * brief:
     *   Handle bank id.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const bankId = exemplarRetrieval.value?.bankId ?? segmentation.value?.retrievalBankId ?? exemplarDecision.value?.bankId ?? 'default-bank'
    feedbackSubmittingFor.value = exemplarId
    try {
      /**
       * brief:
       *   Handle result.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const result = await sendExemplarFeedback({
        exemplarId,
        bankId,
        failureMode,
        qualityScore: reportResult.value ? 0.85 : undefined,
        uncertainty: segmentation.value?.retrievalUncertainty ?? undefined,
        metadata: {
          imageFilename: uploadedImage.value?.filename ?? '',
          patientId: patient.value.patientId,
        },
      })
      exemplarFeedback.value = {
        ...exemplarFeedback.value,
        [exemplarId]: result,
      }
      pushToast(`Exemplar feedback saved: ${failureMode}`, 'success')
    } catch (error) {
      /**
       * brief:
       *   Handle message.
       *
       * parameter:
       *   - None.
       *
       * retrival:
       *   - Returns the computed value or updates local application state.
       */
      const message = resolveRequestErrorMessage(error, 'Exemplar feedback failed.')
      pushToast(message, 'error')
    } finally {
      feedbackSubmittingFor.value = null
    }
  }

  /**
   * brief:
   *   Toggle mask.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const toggleMask = () => {
    showMask.value = !showMask.value
  }

  /**
   * brief:
   *   Dispose .
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const dispose = () => {
    revokeCurrentObjectUrl()
  }

  return {
    patient,
    expertConfig,
    uploadedImage,
    uploadedMaskFile,
    segmentation,
    reportResult,
    exemplarDecision,
    exemplarRetrieval,
    exemplarFeedback,
    showMask,
    isSegmenting,
    isGeneratingReport,
    isEvaluatingExemplar,
    isRetrievingExemplars,
    feedbackSubmittingFor,
    toast,
    canSegment,
    canGenerateReport,
    canEvaluateExemplar,
    updatePatient,
    ingestLocalImage,
    ingestMaskImage,
    applyUploadedMask,
    runSegmentation,
    updateExpertConfig,
    generateReport,
    evaluateExemplar,
    refreshExemplarRetrieval,
    submitExemplarFeedback,
    toggleMask,
    dispose,
  }
})
