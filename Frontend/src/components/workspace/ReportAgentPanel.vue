<script setup lang="ts">
import MarkdownReportViewer from '@/components/workspace/MarkdownReportViewer.vue'
import type { AgentTraceStep, FeatureTagTone, WorkspaceFeatureTag, WorkspaceReportResult } from '@/types/workspace'

defineProps<{
  reportResult: WorkspaceReportResult | null
  isGenerating: boolean
  canGenerate: boolean
}>()

defineEmits<{
  (event: 'generate'): void
}>()

const toneClasses: Record<FeatureTagTone, string> = {
  sky: 'bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200',
  emerald: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-200',
  amber: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200',
  rose: 'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-200',
  violet: 'bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-200',
}

const resolveTagTone = (tag: WorkspaceFeatureTag) => toneClasses[tag.tone] ?? toneClasses.sky

const traceToneClasses: Record<AgentTraceStep['kind'], string> = {
  thought: 'border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900/60 dark:bg-sky-950/30 dark:text-sky-200',
  tool_call: 'border-violet-200 bg-violet-50 text-violet-800 dark:border-violet-900/60 dark:bg-violet-950/30 dark:text-violet-200',
  tool_result: 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-200',
  final: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-100',
}

const resolveTraceTone = (step: AgentTraceStep) => traceToneClasses[step.kind]
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex flex-col gap-4 border-b border-slate-200 pb-4 dark:border-slate-700 xl:flex-row xl:items-end xl:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">报告 Agent</h3>
      </div>

      <button
        type="button"
        class="surface-button-primary px-4 py-3"
        :disabled="!canGenerate || isGenerating"
        @click="$emit('generate')"
      >
        {{ isGenerating ? '生成中...' : '生成正式诊断报告' }}
      </button>
    </div>

    <div v-if="reportResult" class="mt-5 grid gap-4">
      <div class="grid gap-3 md:grid-cols-3">
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
      </div>

      <article class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="text-sm font-semibold text-slate-900 dark:text-white">病例特征索引标签</p>
          <span class="surface-badge bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200">
            {{ reportResult.featureTags.length }} 个标签
          </span>
        </div>

        <div class="mt-4 flex flex-wrap gap-2">
          <span
            v-for="tag in reportResult.featureTags"
            :key="tag.id"
            class="rounded-full px-3 py-1.5 text-sm font-medium"
            :class="resolveTagTone(tag)"
          >
            {{ tag.label }}
          </span>
        </div>
      </article>

      <MarkdownReportViewer :markdown="reportResult.reportMarkdown" />

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

      <article class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
        <div class="flex items-center justify-between gap-3">
          <p class="text-sm font-semibold text-slate-900 dark:text-white">Agent 思考与工具调用轨迹</p>
          <span class="surface-badge bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            {{ reportResult.agentTrace.length }} steps
          </span>
        </div>

        <div class="mt-4 grid gap-3">
          <article
            v-for="step in reportResult.agentTrace"
            :key="step.id"
            class="rounded-3xl border px-4 py-3"
            :class="resolveTraceTone(step)"
          >
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p class="text-sm font-semibold">{{ step.title }}</p>
                <p class="mt-1 text-xs uppercase tracking-[0.14em] opacity-80">
                  {{ step.kind }}
                  <span v-if="step.toolName">· {{ step.toolName }}</span>
                  <span v-if="step.status">· {{ step.status }}</span>
                </p>
              </div>
            </div>
            <p class="mt-3 text-sm leading-6 opacity-90">{{ step.detail }}</p>
          </article>
        </div>
      </article>
    </div>

    <div
      v-else
      class="mt-5 rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    />
  </section>
</template>
