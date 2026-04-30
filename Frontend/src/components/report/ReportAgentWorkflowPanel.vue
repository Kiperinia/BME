<script setup lang="ts">
import { computed } from 'vue'

import type { AgentWorkflowSummary } from '@/types/eis'

const props = defineProps<{
  workflow: AgentWorkflowSummary | null
}>()

const workflowStateLabel = computed(() => {
  if (!props.workflow) {
    return '待运行'
  }

  return props.workflow.workflowMode === 'llm' ? 'LLM Agent' : '规则 Agent'
})

const workflowGeneratedAt = computed(() => {
  if (!props.workflow) {
    return '尚未运行'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(props.workflow.generatedAt))
})

const formatDisposition = (value: string) => value.replaceAll('_', ' ')
</script>

<template>
  <section class="surface-card flex min-h-0 flex-col p-4">
    <div class="flex items-start justify-between gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">Agent 工作流输出</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          展示 Agent 主流程、SAM3 分割接入结果与病灶级推理摘要。
        </p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200">
        {{ workflowStateLabel }}
      </span>
    </div>

    <div v-if="!workflow" class="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400">
      尚未触发 Agent 工作流。生成草稿或标签后，这里会展示分割、分型、风险和主病灶摘要。
    </div>

    <template v-else>
      <div class="mt-4 grid gap-3 sm:grid-cols-3">
        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">病灶数量</p>
          <p class="mt-1 text-lg font-semibold text-slate-900 dark:text-white">{{ workflow.lesionCount }}</p>
        </article>

        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">主病灶</p>
          <p class="mt-1 text-sm font-semibold text-slate-900 dark:text-white">{{ workflow.highestRiskLesionId ?? '待确认' }}</p>
        </article>

        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">生成时间</p>
          <p class="mt-1 text-sm font-semibold text-slate-900 dark:text-white">{{ workflowGeneratedAt }}</p>
        </article>
      </div>

      <div class="mt-4 rounded-2xl bg-slate-50 px-4 py-3 text-xs text-slate-600 dark:bg-slate-900 dark:text-slate-300 md:text-sm">
        <p class="font-medium text-slate-900 dark:text-white">{{ workflow.agentName }}</p>
        <p class="mt-1">{{ workflow.pipeline }}</p>
        <p class="mt-1">模型版本：{{ workflow.modelVersion }}</p>
      </div>

      <div class="mt-4">
        <h4 class="text-sm font-semibold text-slate-900 dark:text-white">流程步骤</h4>
        <ol class="mt-2 space-y-2 text-xs text-slate-600 dark:text-slate-300 md:text-sm">
          <li
            v-for="(step, index) in workflow.steps"
            :key="`${workflow.generatedAt}-${index}`"
            class="rounded-xl bg-slate-50 px-3 py-2 dark:bg-slate-900"
          >
            {{ index + 1 }}. {{ step }}
          </li>
        </ol>
      </div>

      <div
        v-if="workflow.warnings.length"
        class="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-200 md:text-sm"
      >
        <p class="font-medium">工作流提示</p>
        <p
          v-for="warning in workflow.warnings"
          :key="warning"
          class="mt-1"
        >
          {{ warning }}
        </p>
      </div>

      <div class="mt-4 min-h-0 flex-1 overflow-auto pr-1">
        <div class="grid gap-3">
          <article
            v-for="lesion in workflow.lesions"
            :key="lesion.lesionId"
            class="rounded-2xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-900"
          >
            <div class="flex items-center justify-between gap-3">
              <div>
                <h5 class="text-sm font-semibold text-slate-900 dark:text-white">{{ lesion.lesionId }}</h5>
                <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">{{ lesion.sourceLabel }} · {{ lesion.label }}</p>
              </div>
              <span class="rounded-full bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-700 dark:bg-sky-950/70 dark:text-sky-200">
                {{ (lesion.confidence * 100).toFixed(0) }}%
              </span>
            </div>

            <div class="mt-3 grid gap-2 text-xs text-slate-600 dark:text-slate-300 sm:grid-cols-2">
              <p>Paris 分型：{{ lesion.parisType }}</p>
              <p>浸润风险：{{ lesion.invasionRisk }}</p>
              <p>综合风险：{{ lesion.riskLevel }} / {{ lesion.totalScore.toFixed(1) }}</p>
              <p>建议处置：{{ formatDisposition(lesion.disposition) }}</p>
              <p>估计大小：{{ lesion.estimatedSizeMm.toFixed(1) }} mm</p>
              <p>LLM 参与：{{ lesion.usedLlm ? '是' : '否' }}</p>
            </div>

            <p v-if="lesion.shapeDescription" class="mt-3 text-xs text-slate-500 dark:text-slate-400 md:text-sm">
              {{ lesion.shapeDescription }}
            </p>
          </article>
        </div>
      </div>
    </template>
  </section>
</template>