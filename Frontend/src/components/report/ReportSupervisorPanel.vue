<script setup lang="ts">
import type { SupervisorDecision, SupervisorIssue } from '@/types/supervisor'

defineProps<{
  decision: SupervisorDecision | null
}>()

const statusLabels: Record<string, string> = {
  approved: '通过',
  rejected: '拒绝',
  human_review: '人工复核',
  failed: '失败',
}

const statusClasses: Record<string, string> = {
  approved: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200',
  rejected: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200',
  human_review: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200',
  failed: 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200',
}

const riskLabels: Record<string, string> = {
  low: '低风险',
  medium: '中风险',
  high: '高风险',
  critical: '极高风险',
}

const severityClasses: Record<string, string> = {
  info: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  warn: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200',
  error: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200',
  critical: 'bg-rose-200 text-rose-900 dark:bg-rose-900/60 dark:text-rose-100',
}

const resolveStatusLabel = (status: string) => statusLabels[status] ?? status
const resolveStatusClass = (status: string) => statusClasses[status] ?? statusClasses.failed
const resolveSeverityClass = (issue: SupervisorIssue) => severityClasses[issue.severity] ?? severityClasses.info

</script>

<template>
  <section class="surface-card flex min-h-0 flex-col p-4">
    <div class="flex items-start justify-between gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">监督结论</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          展示监督规则的判定结果与问题列表。
        </p>
      </div>
      <span
        class="surface-badge"
        :class="decision ? resolveStatusClass(decision.status) : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'"
      >
        {{ decision ? resolveStatusLabel(decision.status) : '未评估' }}
      </span>
    </div>

    <div v-if="!decision" class="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400">
      尚未生成监督结果。
    </div>

    <template v-else>
      <div class="mt-4 grid gap-3 sm:grid-cols-2">
        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">风险等级</p>
          <p class="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
            {{ riskLabels[decision.riskLevel] ?? decision.riskLevel }}
          </p>
        </article>
        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">硬例标记</p>
          <p class="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
            {{ decision.hardCase ? '是' : '否' }}
          </p>
        </article>
      </div>

      <div v-if="decision.routing.length" class="mt-3 flex flex-wrap gap-2">
        <span
          v-for="route in decision.routing"
          :key="route"
          class="rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold text-sky-700 dark:bg-sky-900/40 dark:text-sky-200"
        >
          路由：{{ route }}
        </span>
      </div>

      <div class="mt-4">
        <div class="flex items-center justify-between gap-3">
          <h4 class="text-sm font-semibold text-slate-900 dark:text-white">问题列表</h4>
          <span class="text-xs text-slate-500 dark:text-slate-400">
            {{ decision.issues.length }} 条
          </span>
        </div>

        <div v-if="!decision.issues.length" class="mt-3 rounded-2xl bg-emerald-50 px-4 py-3 text-xs text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200">
          未发现明显问题。
        </div>

        <div v-else class="mt-3 grid gap-2">
          <article
            v-for="issue in decision.issues"
            :key="`${issue.type}-${issue.message}`"
            class="rounded-2xl border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-900"
          >
            <div class="flex items-center justify-between gap-2">
              <p class="text-xs font-semibold text-slate-900 dark:text-white">{{ issue.type }}</p>
              <span class="rounded-full px-2 py-0.5 text-[10px] font-semibold" :class="resolveSeverityClass(issue)">
                {{ issue.severity }}
              </span>
            </div>
            <p class="mt-1 text-xs text-slate-600 dark:text-slate-300">{{ issue.message }}</p>
            <p v-if="issue.location" class="mt-1 text-[10px] text-slate-500 dark:text-slate-400">
              位置：{{ issue.location }}
            </p>
          </article>
        </div>
      </div>
    </template>
  </section>
</template>
