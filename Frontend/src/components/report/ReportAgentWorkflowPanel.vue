<script setup lang="ts">
import { computed } from 'vue'

import type { AgentDetailSummary, AgentMainTool, AgentRun, AgentWorkflowSummary } from '@/types/eis'

/**
 * brief:
 *   Handle props.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const props = defineProps<{
  workflow: AgentWorkflowSummary | null
}>()

/**
 * brief:
 *   Handle workflow state label.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const workflowStateLabel = computed(() => {
  if (!props.workflow) {
    return '待运行'
  }

  return props.workflow.workflowMode === 'llm' ? 'LLM Agent' : '规则 Agent'
})

/**
 * brief:
 *   Handle workflow generated at.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
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

/**
 * brief:
 *   Handle as main tool chain.
 *
 * parameter:
 *   - value: Input value for value.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const asMainToolChain = (value: unknown): AgentMainTool[] => {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map((item) => ({
      name: String(item.name ?? ''),
      description: String(item.description ?? ''),
    }))
    .filter((item) => item.name)
}

/**
 * brief:
 *   Handle agent details.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const agentDetails = computed<AgentDetailSummary[]>(() => {
  /**
   * brief:
   *   Handle closed loop details.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const closedLoopDetails = props.workflow?.closedLoopSummary?.agentDetails
  if (closedLoopDetails?.length) {
    return closedLoopDetails
  }

  return (props.workflow?.agentRuns ?? []).map((run: AgentRun) => {
    /**
     * brief:
     *   Handle observations.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const observations = run.observations ?? {}
    /**
     * brief:
     *   Handle main tool chain.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const mainToolChain = asMainToolChain(observations.mainToolChain)
    return {
      agentName: run.agentName,
      displayName: run.displayName || run.agentName,
      detail: String(observations.agentDetail ?? run.goal),
      promptDesign: Array.isArray(observations.promptDesign)
        ? observations.promptDesign.map((item) => String(item))
        : [],
      goal: run.goal,
      status: run.status,
      decision: run.decision,
      mainToolChain: mainToolChain.length
        ? mainToolChain
        : run.toolCalls.map((call) => ({
            name: call.tool_name,
            description: call.status,
          })),
      warnings: run.warnings,
      keyOutputs: observations,
    }
  })
})

/**
 * brief:
 *   Format disposition.
 *
 * parameter:
 *   - value: Input value for value.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const formatDisposition = (value: string) => value.replaceAll('_', ' ')

/**
 * brief:
 *   Handle compact output.
 *
 * parameter:
 *   - value: Input value for value.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const compactOutput = (value: unknown) => {
  if (value === null || value === undefined || value === '') {
    return '无'
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return `${value.length} 项`
  }
  try {
    /**
     * brief:
     *   Handle text.
     *
     * parameter:
     *   - outputs: Input value for outputs.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const text = JSON.stringify(value)
    return text.length > 120 ? `${text.slice(0, 120)}...` : text
  } catch {
    return String(value)
  }
}

/**
 * brief:
 *   Handle key output entries.
 *
 * parameter:
 *   - outputs: Input value for outputs.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const keyOutputEntries = (outputs: Record<string, unknown>) => {
  return Object.entries(outputs)
    .filter(([key]) => !['agentDetail', 'promptDesign', 'mainToolChain', 'diagnosis'].includes(key))
    .slice(0, 5)
}
</script>

<template>
  <section class="surface-card flex min-h-0 flex-col p-4">
    <div class="flex items-start justify-between gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">Agent 工作流输出</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          展示分割预处理、样本审核、报告生成、标签嵌入和结果复核的主工具链。
        </p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200">
        {{ workflowStateLabel }}
      </span>
    </div>

    <div v-if="!workflow" class="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400">
      尚未触发 Agent 工作流。
    </div>

    <template v-else>
      <div class="mt-4 grid gap-3 sm:grid-cols-3">
        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">病灶数量</p>
          <p class="mt-1 text-lg font-semibold text-slate-900 dark:text-white">{{ workflow.lesionCount }}</p>
        </article>

        <article class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
          <p class="text-xs text-slate-500 dark:text-slate-400">最终复核</p>
          <p class="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
            {{ workflow.closedLoopSummary?.finalDecision ?? workflow.closedLoopSummary?.finalStatus ?? '待确认' }}
          </p>
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
        <p v-if="workflow.closedLoopSummary" class="mt-1">
          质量分：{{ Number(workflow.closedLoopSummary.qualityScore ?? 0).toFixed(2) }}
          / 数据库词条：{{ workflow.closedLoopSummary.databaseRecordCount ?? 0 }}
        </p>
      </div>

      <div v-if="agentDetails.length" class="mt-4">
        <h4 class="text-sm font-semibold text-slate-900 dark:text-white">智能体细节讲解与主工具链</h4>
        <div class="mt-2 grid gap-3">
          <article
            v-for="agent in agentDetails"
            :key="agent.agentName"
            class="rounded-2xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-900"
          >
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h5 class="text-sm font-semibold text-slate-900 dark:text-white">{{ agent.displayName }}</h5>
                <p class="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400 md:text-sm">{{ agent.detail }}</p>
              </div>
              <span class="surface-badge bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200">
                {{ agent.decision || agent.status }}
              </span>
            </div>

            <div v-if="agent.promptDesign.length" class="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-950">
              <p class="text-xs font-semibold text-slate-900 dark:text-white">Prompt 设计</p>
              <ul class="mt-2 grid gap-1 text-xs leading-5 text-slate-600 dark:text-slate-300 md:text-sm">
                <li
                  v-for="prompt in agent.promptDesign"
                  :key="`${agent.agentName}-${prompt}`"
                >
                  {{ prompt }}
                </li>
              </ul>
            </div>

            <div class="mt-3 grid gap-2">
              <div
                v-for="tool in agent.mainToolChain"
                :key="`${agent.agentName}-${tool.name}`"
                class="rounded-xl bg-slate-50 px-3 py-2 text-xs text-slate-600 dark:bg-slate-950 dark:text-slate-300 md:text-sm"
              >
                <span class="font-semibold text-slate-900 dark:text-white">{{ tool.name }}</span>
                <span v-if="tool.description">：{{ tool.description }}</span>
              </div>
            </div>

            <div v-if="keyOutputEntries(agent.keyOutputs).length" class="mt-3 grid gap-1 text-xs text-slate-500 dark:text-slate-400 md:text-sm">
              <p
                v-for="[key, value] in keyOutputEntries(agent.keyOutputs)"
                :key="`${agent.agentName}-${key}`"
              >
                <span class="font-medium text-slate-700 dark:text-slate-200">{{ key }}</span>：{{ compactOutput(value) }}
              </p>
            </div>

            <div v-if="agent.warnings.length" class="mt-3 grid gap-1">
              <p
                v-for="warning in agent.warnings"
                :key="warning"
                class="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200 md:text-sm"
              >
                {{ warning }}
              </p>
            </div>
          </article>
        </div>
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
                <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">{{ lesion.sourceLabel }} / {{ lesion.label }}</p>
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
