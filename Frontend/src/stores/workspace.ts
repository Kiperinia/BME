import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import {
  evaluateExemplarCandidate,
  generateWorkspaceReport,
  segmentWorkspaceImage,
} from '@/api/workspace'
import {
  createDefaultExpertConfiguration,
  createDefaultPatient,
  type ExemplarBankDecision,
  type ExpertConfiguration,
  type ToastState,
  type UploadedWorkspaceImage,
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

export const useWorkspaceStore = defineStore('workspace', () => {
  const patient = ref(createDefaultPatient())
  const expertConfig = ref(createDefaultExpertConfiguration())
  const uploadedFile = ref<File | null>(null)
  const uploadedImage = ref<UploadedWorkspaceImage | null>(null)
  const segmentation = ref<WorkspaceSegmentation | null>(null)
  const reportResult = ref<WorkspaceReportResult | null>(null)
  const exemplarDecision = ref<ExemplarBankDecision | null>(null)
  const showMask = ref(true)
  const isSegmenting = ref(false)
  const isGeneratingReport = ref(false)
  const isEvaluatingExemplar = ref(false)
  const toast = ref<ToastState>({
    visible: false,
    message: '',
    tone: 'info',
  })

  let toastTimer: number | undefined

  const canSegment = computed(() => Boolean(uploadedFile.value && uploadedImage.value))
  const canGenerateReport = computed(() => Boolean(uploadedImage.value && segmentation.value?.maskCoordinates.length))
  const canEvaluateExemplar = computed(() => Boolean(uploadedImage.value && segmentation.value?.maskCoordinates.length))

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
    showMask.value = true
  }

  const updateExpertConfig = (nextValue: ExpertConfiguration) => {
    expertConfig.value = { ...nextValue }
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
      patientId: `case-${file.name.replace(/\.[^.]+$/, '').slice(0, 48) || 'workspace'}`,
    }
    resetWorkflowState()
    pushToast('本地图像已载入，可以开始 MedicalSAM3 分割。', 'success')
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
      })
      segmentation.value = result
      showMask.value = true
      reportResult.value = null
      exemplarDecision.value = null
      pushToast('MedicalSAM3 分割完成。', 'success')
    } catch (error) {
      const message = error instanceof Error ? error.message : '分割失败，请检查后端服务。'
      pushToast(message, 'error')
    } finally {
      isSegmenting.value = false
    }
  }

  const buildReportRequest = (): WorkspaceReportRequest | null => {
    if (!uploadedImage.value || !segmentation.value) {
      return null
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
      segmentation: segmentation.value,
      expertConfig: expertConfig.value,
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
      pushToast('诊断报告已生成，请医生复核。', 'success')
    } catch (error) {
      const message = error instanceof Error ? error.message : '报告生成失败。'
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
      const message = error instanceof Error ? error.message : '样本库评估失败。'
      pushToast(message, 'error')
    } finally {
      isEvaluatingExemplar.value = false
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
    segmentation,
    reportResult,
    exemplarDecision,
    showMask,
    isSegmenting,
    isGeneratingReport,
    isEvaluatingExemplar,
    toast,
    canSegment,
    canGenerateReport,
    canEvaluateExemplar,
    ingestLocalImage,
    runSegmentation,
    updateExpertConfig,
    generateReport,
    evaluateExemplar,
    toggleMask,
    dispose,
  }
})
