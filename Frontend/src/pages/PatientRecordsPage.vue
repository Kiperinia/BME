<script setup lang="ts">
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'

import MarkdownReportViewer from '@/components/workspace/MarkdownReportViewer.vue'
import { usePatientRecordsStore } from '@/stores/patientRecords'
import type { FeatureTagTone, WorkspaceFeatureTag } from '@/types/workspace'

const patientRecordsStore = usePatientRecordsStore()
const { records, selectedRecord } = storeToRefs(patientRecordsStore)
const activeFilterLabels = ref<string[]>([])

const toneClasses: Record<FeatureTagTone, string> = {
  sky: 'bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200',
  emerald: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-200',
  amber: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200',
  rose: 'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-200',
  violet: 'bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-200',
}

const resolveTagTone = (tag: WorkspaceFeatureTag) => toneClasses[tag.tone] ?? toneClasses.sky

const availableFilterTags = computed(() => {
  const map = new Map<string, WorkspaceFeatureTag>()
  for (const record of records.value) {
    for (const tag of record.featureTags) {
      if (!map.has(tag.label)) {
        map.set(tag.label, tag)
      }
    }
  }
  return Array.from(map.values())
})

const filteredRecords = computed(() => {
  if (!activeFilterLabels.value.length) {
    return records.value
  }

  return records.value.filter((record) => {
    const labelSet = new Set(record.featureTags.map((tag) => tag.label))
    return activeFilterLabels.value.every((label) => labelSet.has(label))
  })
})

const displayedSelectedRecord = computed(
  () => filteredRecords.value.find((record) => record.recordId === selectedRecord.value?.recordId) ?? null,
)

const formattedRecordCount = computed(() => {
  if (!activeFilterLabels.value.length) {
    return `${records.value.length} 次诊断记录`
  }
  return `${filteredRecords.value.length}/${records.value.length} 条匹配`
})

const toggleFilterTag = (label: string) => {
  if (activeFilterLabels.value.includes(label)) {
    activeFilterLabels.value = activeFilterLabels.value.filter((item) => item !== label)
    return
  }

  activeFilterLabels.value = [...activeFilterLabels.value, label]
}

const clearFilters = () => {
  activeFilterLabels.value = []
}

const closePreview = () => {
  patientRecordsStore.selectRecord('')
}
</script>

<template>
  <main class="mx-auto flex min-h-[calc(100vh-88px)] w-full max-w-[1880px] flex-col gap-4 px-4 py-4 lg:px-6">
    <section class="surface-card p-6">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 class="text-2xl font-semibold text-slate-900 dark:text-white">病例特征索引页</h2>
        </div>
        <div class="flex flex-wrap items-center justify-end gap-2">
          <button
            type="button"
            class="surface-button-secondary px-3 py-2 text-sm"
            @click="patientRecordsStore.createRandomRecords(8)"
          >
            一键随机生成实体
          </button>
          <button
            type="button"
            class="surface-button-secondary px-3 py-2 text-sm"
            :disabled="!records.length"
            @click="patientRecordsStore.clearAllRecords()"
          >
            清除所有患者
          </button>
          <button
            type="button"
            class="surface-button-secondary px-3 py-2 text-sm"
            :disabled="!activeFilterLabels.length"
            @click="clearFilters"
          >
            清空筛选
          </button>
          <span class="surface-badge bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            {{ formattedRecordCount }}
          </span>
        </div>
      </div>

      <div v-if="availableFilterTags.length" class="mt-5 rounded-3xl border border-slate-200 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/60">
        <div class="mb-3 flex items-center justify-between gap-2">
          <p class="text-sm font-semibold text-slate-800 dark:text-slate-100">标签筛选</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="tag in availableFilterTags"
            :key="`filter-${tag.label}`"
            type="button"
            class="rounded-full border px-3 py-1.5 text-xs font-medium transition"
            :class="[
              resolveTagTone(tag),
              activeFilterLabels.includes(tag.label)
                ? 'ring-2 ring-sky-300 dark:ring-sky-800 border-sky-400 dark:border-sky-700'
                : 'border-transparent opacity-85 hover:opacity-100',
            ]"
            @click="toggleFilterTag(tag.label)"
          >
            {{ tag.label }}
          </button>
        </div>
      </div>
    </section>

    <section
      v-if="records.length"
      class="grid items-start gap-4"
      :class="displayedSelectedRecord ? '2xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)]' : ''"
    >
      <aside class="grid content-start gap-4">
        <div v-if="filteredRecords.length" class="columns-1 gap-4 md:columns-2 xl:columns-3 2xl:columns-4">
          <button
            v-for="record in filteredRecords"
            :key="record.recordId"
            type="button"
            class="surface-card mb-4 w-full break-inside-avoid p-5 text-left transition hover:border-sky-300 dark:hover:border-sky-700"
            :class="displayedSelectedRecord?.recordId === record.recordId ? 'border-sky-400 ring-2 ring-sky-200 dark:border-sky-600 dark:ring-sky-900/50' : ''"
            @click="patientRecordsStore.selectRecord(record.recordId)"
          >
            <div class="flex items-start justify-between gap-3">
              <div>
                <p class="text-base font-semibold text-slate-900 dark:text-white">{{ record.patient.patientName || '未填写姓名' }}</p>
                <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
                  {{ record.patient.patientId }} · {{ record.patient.examDate || '未填写日期' }}
                </p>
              </div>
              <span class="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                {{ new Date(record.createdAt).toLocaleString('zh-CN') }}
              </span>
            </div>

            <div class="mt-4 flex flex-wrap gap-2">
              <span
                v-for="tag in record.featureTags"
                :key="tag.id"
                class="rounded-full px-3 py-1.5 text-xs font-medium"
                :class="resolveTagTone(tag)"
              >
                {{ tag.label }}
              </span>
            </div>

            <div class="mt-4 grid gap-2 text-sm text-slate-600 dark:text-slate-300">
              <p>Paris：{{ record.parisClassification || '未填写' }}</p>
              <p>类型：{{ record.lesionType || '未填写' }}</p>
              <p>风险：{{ record.riskLevel || '未记录' }}</p>
            </div>
          </button>
        </div>

        <article
          v-else
          class="surface-card rounded-3xl border border-dashed border-slate-300 p-5 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400"
        >
          暂无匹配病例
        </article>
      </aside>

      <section v-if="displayedSelectedRecord" class="grid gap-4">
        <article class="surface-card p-5">
          <div class="flex flex-col gap-3 border-b border-slate-200 pb-4 dark:border-slate-700 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <h3 class="text-xl font-semibold text-slate-900 dark:text-white">{{ displayedSelectedRecord.patient.patientName || '未填写姓名' }}</h3>
              <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
                {{ displayedSelectedRecord.patient.patientId }} · {{ displayedSelectedRecord.imageFilename }} · {{ displayedSelectedRecord.workflowMode }}
              </p>
            </div>
            <div class="flex items-center gap-2">
              <span class="surface-badge bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-200">
                {{ displayedSelectedRecord.featureTags.length }} 个特征索引标签
              </span>
              <button
                type="button"
                class="surface-button-secondary px-3 py-2 text-sm"
                @click="closePreview"
              >
                收起预览
              </button>
            </div>
          </div>

          <div class="mt-4 grid gap-3 md:grid-cols-3">
            <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
              <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Findings</p>
              <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ displayedSelectedRecord.findings }}</p>
            </article>
            <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
              <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Conclusion</p>
              <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ displayedSelectedRecord.conclusion }}</p>
            </article>
            <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
              <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Recommendation</p>
              <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ displayedSelectedRecord.recommendation }}</p>
            </article>
          </div>
        </article>

        <article class="surface-card p-5">
          <div class="flex flex-wrap gap-2">
            <span
              v-for="tag in displayedSelectedRecord.featureTags"
              :key="tag.id"
              class="rounded-full px-3 py-1.5 text-sm font-medium"
              :class="resolveTagTone(tag)"
            >
              {{ tag.label }}
            </span>
          </div>
        </article>

        <MarkdownReportViewer :markdown="displayedSelectedRecord.reportMarkdown" />
      </section>
    </section>

    <section
      v-else
      class="surface-card flex min-h-[320px] items-center justify-center p-6 text-center text-sm text-slate-500 dark:text-slate-400"
    >
      暂无病例记录
    </section>
  </main>
</template>
