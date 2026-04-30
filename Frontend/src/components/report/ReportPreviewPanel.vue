<script setup lang="ts">
import { computed } from 'vue'

import type { PatientRecord } from '@/types/eis'

const props = defineProps<{
  patient: PatientRecord
  savedAtLabel: string
  findings: string
  conclusion: string
  layoutSuggestion: string
  annotationCount: number
}>()

const examDateLabel = computed(() => {
  if (!props.patient.examDate) {
    return '未记录'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(new Date(props.patient.examDate))
})

const findingsLabel = computed(() => props.findings.trim() || '待生成检查所见')
const conclusionLabel = computed(() => props.conclusion.trim() || '待生成诊断结论')
const layoutSuggestionLabel = computed(() => props.layoutSuggestion.trim() || '暂未提供排版建议')
const statusLabel = computed(() => {
  switch (props.patient.status) {
    case 0:
      return '待处理'
    case 1:
      return '草稿中'
    case 2:
      return '已完成'
    default:
      return '未知状态'
  }
})
</script>

<template>
  <article class="surface-card flex h-full min-h-0 flex-col overflow-hidden bg-white shadow-[0_18px_45px_rgba(15,23,42,0.12)] dark:bg-slate-900">
    <div class="border-b border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-950/80">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.28em] text-slate-500 dark:text-slate-400">
            EIS Report Preview
          </p>
          <h3 class="mt-1.5 text-lg font-semibold text-slate-900 dark:text-white">内镜检查报告预览</h3>
        </div>
        <div class="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
          {{ statusLabel }}
        </div>
      </div>
    </div>

    <div class="mx-auto flex min-h-0 w-full max-w-[794px] flex-1 flex-col overflow-hidden bg-[linear-gradient(180deg,#ffffff_0%,#f8fafc_100%)] px-5 py-4 text-slate-800 dark:bg-[linear-gradient(180deg,#0f172a_0%,#111827_100%)] dark:text-slate-100">
      <header class="border-b border-dashed border-slate-300 pb-4 dark:border-slate-700">
        <div class="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p class="text-[11px] uppercase tracking-[0.3em] text-slate-400">Endoscopy Intelligence Suite</p>
            <h4 class="mt-2 text-2xl font-semibold text-slate-950 dark:text-white">电子内镜检查报告</h4>
          </div>
          <div class="rounded-2xl border border-slate-200 px-3 py-2 text-xs leading-5 dark:border-slate-700 md:text-sm md:leading-6">
            <p>保存时间：{{ savedAtLabel }}</p>
            <p>智能标签：{{ annotationCount }} 项</p>
          </div>
        </div>
      </header>

      <section class="mt-4 grid gap-2 rounded-2xl border border-slate-200 p-3 text-xs leading-5 dark:border-slate-700 md:grid-cols-2 md:text-sm md:leading-6">
        <div>
          <span class="text-slate-500 dark:text-slate-400">患者姓名：</span>
          <span class="font-medium text-slate-900 dark:text-white">{{ patient.patientName }}</span>
        </div>
        <div>
          <span class="text-slate-500 dark:text-slate-400">患者编号：</span>
          <span class="font-medium text-slate-900 dark:text-white">{{ patient.patientId }}</span>
        </div>
        <div>
          <span class="text-slate-500 dark:text-slate-400">性别 / 年龄：</span>
          <span class="font-medium text-slate-900 dark:text-white">{{ patient.gender }} / {{ patient.age }} 岁</span>
        </div>
        <div>
          <span class="text-slate-500 dark:text-slate-400">检查日期：</span>
          <span class="font-medium text-slate-900 dark:text-white">{{ examDateLabel }}</span>
        </div>
      </section>

      <section class="mt-4 flex min-h-0 flex-1 flex-col gap-4 text-sm leading-6 md:text-[15px] md:leading-7">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Findings</p>
          <div class="mt-2 rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-800/70">
            {{ findingsLabel }}
          </div>
        </div>

        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Conclusion</p>
          <div class="mt-2 rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-800/70">
            {{ conclusionLabel }}
          </div>
        </div>

        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Layout Suggestion</p>
          <div class="mt-2 rounded-2xl border border-dashed border-slate-300 px-4 py-3 text-slate-600 dark:border-slate-600 dark:text-slate-300">
            {{ layoutSuggestionLabel }}
          </div>
        </div>
      </section>

      <footer class="mt-4 border-t border-dashed border-slate-300 pt-3 text-[11px] leading-5 text-slate-500 dark:border-slate-700 dark:text-slate-400 md:text-xs md:leading-6">
        本预览用于展示 Agent 草稿和医生编辑结果，最终出具版本可继续接入打印、导出或签名流程。
      </footer>
    </div>
  </article>
</template>