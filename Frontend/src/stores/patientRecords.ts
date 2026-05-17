import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'

import type { PatientCaseRecord } from '@/types/workspace'

const STORAGE_KEY = 'bme-patient-case-records'

type NewCaseRecordInput = Omit<PatientCaseRecord, 'recordId' | 'createdAt'>

const buildFormalPatientId = () => {
  const now = new Date()
  const yyyy = now.getFullYear()
  const mm = String(now.getMonth() + 1).padStart(2, '0')
  const dd = String(now.getDate()).padStart(2, '0')
  const suffix = Math.floor(Math.random() * 10000).toString().padStart(4, '0')
  return `PAT-${yyyy}${mm}${dd}-${suffix}`
}

const SAMPLE_PATIENT_NAMES = ['张伟', '李娜', '王磊', '赵敏', '陈涛', '刘佳', '周倩', '吴晨']
const SAMPLE_FINDINGS = [
  '病灶边界相对清晰，局部血管纹理紊乱，提示需重点复核。',
  '病变区域以平坦型形态为主，表面可见轻度颗粒样改变。',
  '病灶中央轻度凹陷，周缘见浅表充血，建议结合病理判断。',
  '局灶区域黏膜结构破坏，伴不规则血管征象，建议完善随访。',
]
const SAMPLE_CONCLUSIONS = [
  '综合评估为中等风险病灶，建议内镜医师复核后确定处置策略。',
  '倾向低风险病灶，建议按规范进行常规随访。',
  '提示存在浸润相关风险征象，建议优先进一步评估。',
]
const SAMPLE_RECOMMENDATIONS = [
  '建议择期内镜下完整切除并送检。',
  '建议进行高倍放大观察并记录表面微结构。',
  '建议结合病理分型和临床信息进行综合决策。',
]
const SAMPLE_LESION_TYPES = ['adenoma', 'serrated', 'LST', 'hyperplastic']
const SAMPLE_PATHOLOGY = ['低级别上皮内瘤变', '高级别上皮内瘤变', '未明确']
const SAMPLE_WORKFLOW_MODES = ['llm', 'rule-only']
const SAMPLE_PARIS = ['平坦型 / 0-IIb 完全平坦', '隆起型 / 0-IIa 轻微隆起', '凹陷型 / 0-IIc 轻微凹陷']

const SAMPLE_TAG_POOL = [
  { label: '0-IIb', category: 'Paris 分型', tone: 'sky' },
  { label: '完全平坦', category: 'Paris 形态', tone: 'sky' },
  { label: '轻微隆起', category: '形态关键词', tone: 'violet' },
  { label: '血管异常', category: '表面征象', tone: 'amber' },
  { label: '充血', category: '表面征象', tone: 'rose' },
  { label: '浸润风险', category: '风险征象', tone: 'rose' },
  { label: '局灶掩码面积', category: '分割特征', tone: 'sky' },
  { label: '较大掩码面积', category: '分割特征', tone: 'amber' },
  { label: 'endoscopic_resection', category: '处置建议', tone: 'amber' },
  { label: 'intermediate', category: '风险等级', tone: 'rose' },
  { label: 'low', category: '风险等级', tone: 'emerald' },
  { label: 'high', category: '风险等级', tone: 'rose' },
]

const randomPick = <T>(items: T[]): T => items[Math.floor(Math.random() * items.length)] as T

const randomSample = <T>(items: T[], count: number): T[] => {
  const pool = [...items]
  const result: T[] = []
  const limitedCount = Math.min(count, pool.length)

  for (let index = 0; index < limitedCount; index += 1) {
    const chosenIndex = Math.floor(Math.random() * pool.length)
    const chosen = pool.splice(chosenIndex, 1)[0]
    if (chosen !== undefined) {
      result.push(chosen)
    }
  }

  return result
}

const loadRecords = (): PatientCaseRecord[] => {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw) as PatientCaseRecord[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export const usePatientRecordsStore = defineStore('patient-records', () => {
  const records = ref<PatientCaseRecord[]>(loadRecords())
  const selectedRecordId = ref<string>('')

  const selectedRecord = computed(() => {
    return records.value.find((record) => record.recordId === selectedRecordId.value) ?? null
  })

  const addRecord = (payload: NewCaseRecordInput) => {
    const nextRecord: PatientCaseRecord = {
      ...payload,
      recordId: `record-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: new Date().toISOString(),
    }

    records.value = [nextRecord, ...records.value].slice(0, 100)
    selectedRecordId.value = nextRecord.recordId
  }

  const createRandomRecord = () => {
    const patientName = randomPick(SAMPLE_PATIENT_NAMES)
    const riskLevel = randomPick(['low', 'intermediate', 'high'])
    const tagCount = 6 + Math.floor(Math.random() * 4)
    const chosenTags = randomSample(SAMPLE_TAG_POOL, tagCount).map((tag, index) => ({
      id: `${tag.category}-${tag.label}-${Date.now()}-${index}`,
      label: tag.label,
      category: tag.category,
      tone: tag.tone as 'sky' | 'emerald' | 'amber' | 'rose' | 'violet',
    }))

    addRecord({
      patient: {
        patientId: buildFormalPatientId(),
        patientName,
        examDate: new Date(Date.now() - Math.floor(Math.random() * 30) * 86400000).toISOString().slice(0, 10),
      },
      imageFilename: `capture_${Math.random().toString(36).slice(2, 8)}.png`,
      findings: randomPick(SAMPLE_FINDINGS),
      conclusion: randomPick(SAMPLE_CONCLUSIONS),
      recommendation: randomPick(SAMPLE_RECOMMENDATIONS),
      reportMarkdown: `# ${patientName} 诊断报告\n\n- 风险等级：${riskLevel}\n- 诊断要点：${randomPick(SAMPLE_FINDINGS)}\n- 建议：${randomPick(SAMPLE_RECOMMENDATIONS)}`,
      featureTags: chosenTags,
      parisClassification: randomPick(SAMPLE_PARIS),
      lesionType: randomPick(SAMPLE_LESION_TYPES),
      pathologyClassification: randomPick(SAMPLE_PATHOLOGY),
      workflowMode: randomPick(SAMPLE_WORKFLOW_MODES),
      riskLevel,
    })
  }

  const createRandomRecords = (count = 6) => {
    const safeCount = Math.max(1, Math.min(20, Math.floor(count)))
    for (let index = 0; index < safeCount; index += 1) {
      createRandomRecord()
    }
  }

  const selectRecord = (recordId: string) => {
    selectedRecordId.value = recordId
  }

  const clearAllRecords = () => {
    records.value = []
    selectedRecordId.value = ''
  }

  watch(
    records,
    (nextValue) => {
      if (typeof window === 'undefined') {
        return
      }

      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextValue))
    },
    { deep: true },
  )

  return {
    records,
    selectedRecordId,
    selectedRecord,
    addRecord,
    createRandomRecord,
    createRandomRecords,
    selectRecord,
    clearAllRecords,
  }
})
