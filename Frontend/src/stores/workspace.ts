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

const readFileAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(new Error('failed to read local image'))
    reader.readAsDataURL(file)
  })

const measureImage = (src: string) =>
  new Promise<{ width: number; height: number }>((resolve, reject) => {
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

const resolveRequestErrorMessage = (error: unknown, fallbackMessage: string) => {
  if (axios.isAxiosError(error)) {
    const apiMessage = error.response?.data?.message
    if (typeof apiMessage === 'string' && apiMessage.trim()) {
      return apiMessage
    }

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

export const useWorkspaceStore = defineStore('workspace', () => {
  const patientRecordsStore = usePatientRecordsStore()
  const patient = ref(createDefaultPatient())
  const expertConfig = ref(createDefaultExpertConfiguration())
  const uploadedFile = ref<File | null>(null)
  const uploadedMaskFile = ref<File | null>(null)
  const uploadedImage = ref<UploadedWorkspaceImage | null>(null)
  const segmentation = ref<WorkspaceSegmentation | null>(null)
  const reportResult = ref<WorkspaceReportResult | null>(null)
  const exemplarDecision = ref<ExemplarBankDecision | null>(null)
  const exemplarRetrieval = ref<ExemplarRetrievalResult | null>(null)
  const exemplarFeedback = ref<Record<string, ExemplarFeedbackResult>>({})
  const showMask = ref(true)
  const isSegmenting = ref(false)
  const isGeneratingReport = ref(false)
  const isEvaluatingExemplar = ref(false)
  const isRetrievingExemplars = ref(false)
  const feedbackSubmittingFor = ref<string | null>(null)
  const toast = ref<ToastState>({
    visible: false,
    message: '',
    tone: 'info',
  })

  let toastTimer: number | undefined

  const canSegment = computed(() => Boolean(uploadedFile.value && uploadedImage.value))
  const canGenerateReport = computed(() => Boolean(uploadedImage.value))
  const canEvaluateExemplar = computed(() => Boolean(uploadedImage.value && (segmentation.value?.maskDataUrl || segmentation.value?.maskCoordinates.length)))

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

  const resetWorkflowState = () => {
    segmentation.value = null
    reportResult.value = null
    exemplarDecision.value = null
    exemplarRetrieval.value = null
    exemplarFeedback.value = {}
    showMask.value = true
  }

  const updatePatient = (nextValue: WorkspacePatient) => {
    patient.value = {
      ...nextValue,
      patientId: nextValue.patientId.trim() || patient.value.patientId,
    }
  }

  const updateExpertConfig = (nextValue: ExpertConfiguration) => {
    expertConfig.value = {
      ...nextValue,
      parisClassification: formatParisClassification(nextValue.parisDetail),
    }
  }

  const revokeCurrentObjectUrl = () => {
    if (uploadedImage.value?.objectUrl) {
      URL.revokeObjectURL(uploadedImage.value.objectUrl)
    }
  }

  const ingestLocalImage = async (file: File) => {
    revokeCurrentObjectUrl()

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

  const ingestMaskImage = (file: File) => {
    uploadedMaskFile.value = file
    pushToast('掩码图已载入，可点击“应用掩码展示”。', 'success')
  }

  const parseMaskDataFromImage = async (
    imageSource: string,
    dimensions: { width: number; height: number },
  ): Promise<WorkspaceSegmentation> => {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const instance = new Image()
      instance.onload = () => resolve(instance)
      instance.onerror = () => reject(new Error('failed to decode uploaded mask image'))
      instance.src = imageSource
    })

    const canvas = document.createElement('canvas')
    canvas.width = dimensions.width
    canvas.height = dimensions.height
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('failed to build mask parser canvas context')
    }

    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data
    const overlay = context.createImageData(canvas.width, canvas.height)
    const overlayData = overlay.data

    let borderLuminanceSum = 0
    let borderPixelCount = 0
    let transparentPixelCount = 0
    let brightPixelCount = 0
    let darkPixelCount = 0

    for (let index = 0; index < data.length; index += 4) {
      const pixelIndex = index / 4
      const x = pixelIndex % canvas.width
      const y = Math.floor(pixelIndex / canvas.width)
      const r = data[index] ?? 0
      const g = data[index + 1] ?? 0
      const b = data[index + 2] ?? 0
      const alpha = data[index + 3] ?? 0
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

    const borderMeanLuminance = borderPixelCount > 0 ? borderLuminanceSum / borderPixelCount : 0
    const backgroundIsBright = borderMeanLuminance >= 128
    const hasTransparentMask = transparentPixelCount > data.length / 4 * 0.01
    const minorityIsBright = brightPixelCount > 0 && brightPixelCount <= darkPixelCount

    const paintOverlay = (predicate: (luminance: number, alpha: number) => boolean) => {
      let minX = canvas.width
      let minY = canvas.height
      let maxX = 0
      let maxY = 0
      let foregroundCount = 0

      for (let index = 0; index < data.length; index += 4) {
        const pixelIndex = index / 4
        const x = pixelIndex % canvas.width
        const y = Math.floor(pixelIndex / canvas.width)
        const r = data[index] ?? 0
        const g = data[index + 1] ?? 0
        const b = data[index + 2] ?? 0
        const alpha = data[index + 3] ?? 0
        const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
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

    const primaryResult = paintOverlay((luminance, alpha) => {
      if (hasTransparentMask) {
        return alpha > 20
      }
      return backgroundIsBright ? luminance < 200 : luminance > 55
    })

    const pixelCount = Math.max(canvas.width * canvas.height, 1)
    const primaryRatio = primaryResult.foregroundCount / pixelCount
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

    const overlayCanvas = document.createElement('canvas')
    overlayCanvas.width = canvas.width
    overlayCanvas.height = canvas.height
    const overlayContext = overlayCanvas.getContext('2d')
    if (!overlayContext) {
      throw new Error('failed to build mask overlay canvas context')
    }
    overlayContext.putImageData(overlay, 0, 0)

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

  const applyUploadedMask = async () => {
    if (!uploadedMaskFile.value || !uploadedImage.value) {
      pushToast('请先上传原图和掩码图。', 'error')
      return
    }

    try {
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
      const message = resolveRequestErrorMessage(error, '掩码图解析失败，请更换文件后重试。')
      pushToast(message, 'error')
    }
  }

  const runSegmentation = async () => {
    if (!uploadedFile.value || !uploadedImage.value) {
      pushToast('请先选择一张本地图像。', 'error')
      return
    }

    isSegmenting.value = true
    try {
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
      const message = resolveRequestErrorMessage(error, '分割失败，请检查后端服务。')
      pushToast(message, 'error')
    } finally {
      isSegmenting.value = false
    }
  }

  const buildReportRequest = (): WorkspaceReportRequest | null => {
    if (!uploadedImage.value) {
      return null
    }

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

  const refreshExemplarRetrieval = async () => {
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
      const message = resolveRequestErrorMessage(error, 'Exemplar retrieval failed.')
      pushToast(message, 'error')
    } finally {
      isRetrievingExemplars.value = false
    }
  }

  const generateReport = async () => {
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
      const message = resolveRequestErrorMessage(error, '报告生成失败。')
      pushToast(message, 'error')
    } finally {
      isGeneratingReport.value = false
    }
  }

  const evaluateExemplar = async () => {
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
      const message = resolveRequestErrorMessage(error, '样本库评估失败。')
      pushToast(message, 'error')
    } finally {
      isEvaluatingExemplar.value = false
    }
  }

  const submitExemplarFeedback = async (exemplarId: string, failureMode: ExemplarFeedbackMode) => {
    const bankId = exemplarRetrieval.value?.bankId ?? segmentation.value?.retrievalBankId ?? exemplarDecision.value?.bankId ?? 'default-bank'
    feedbackSubmittingFor.value = exemplarId
    try {
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
      const message = resolveRequestErrorMessage(error, 'Exemplar feedback failed.')
      pushToast(message, 'error')
    } finally {
      feedbackSubmittingFor.value = null
    }
  }

  const toggleMask = () => {
    showMask.value = !showMask.value
  }

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
