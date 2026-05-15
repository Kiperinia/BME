<script setup lang="ts">
import type { WorkspaceReportResult } from '@/types/workspace'

defineProps<{
  reportResult: WorkspaceReportResult | null
  isGenerating: boolean
  canGenerate: boolean
}>()

defineEmits<{
  (event: 'generate'): void
}>()
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex flex-col gap-4 border-b border-slate-200 pb-4 dark:border-slate-700 xl:flex-row xl:items-end xl:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">报告 Agent</h3>
        <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
          基于分割结果与专家配置生成结构化诊断报告，便于医生复核与归档。
        </p>
      </div>

      <button
        type="button"
        class="surface-button-primary px-4 py-3"
        :disabled="!canGenerate || isGenerating"
        @click="$emit('generate')"
      >
        {{ isGenerating ? '生成中...' : '生成诊断报告' }}
      </button>
    </div>

    <div v-if="reportResult" class="mt-5 grid gap-4">
      <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
        <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Findings</p>
        <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ reportResult.findings }}</p>
      </article>

      <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
        <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Conclusion</p>
        <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ reportResult.conclusion }}</p>
      </article>

      <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
        <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Recommendation</p>
        <p class="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{{ reportResult.recommendation }}</p>
      </article>

      <article class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
        <div class="flex items-center justify-between gap-3">
          <p class="text-sm font-semibold text-slate-900 dark:text-white">Agent Workflow</p>
          <span class="surface-badge bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200">
            {{ reportResult.workflow.workflowMode }}
          </span>
        </div>

        <div class="mt-3 grid gap-2">
          <p
            v-for="step in reportResult.workflow.steps"
            :key="step"
            class="rounded-2xl bg-slate-50 px-3 py-2 text-sm text-slate-700 dark:bg-slate-900 dark:text-slate-200"
          >
            {{ step }}
          </p>
        </div>

        <div v-if="reportResult.workflow.warnings.length" class="mt-3 grid gap-2">
          <p
            v-for="warning in reportResult.workflow.warnings"
            :key="warning"
            class="rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200"
          >
            {{ warning }}
          </p>
        </div>
      </article>
    </div>

    <div
      v-else
      class="mt-5 rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    >
      完成分割并填写专家配置后，可在这里生成诊断报告。
    </div>
  </section>
</template>
