<script setup lang="ts">
import type { ExemplarBankDecision } from '@/types/workspace'

defineProps<{
  decision: ExemplarBankDecision | null
  isEvaluating: boolean
  canEvaluate: boolean
}>()

defineEmits<{
  (event: 'evaluate'): void
}>()
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex flex-col gap-4 border-b border-slate-200 pb-4 dark:border-slate-700 xl:flex-row xl:items-end xl:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">Exemplar Bank Agent</h3>
      </div>

      <button
        type="button"
        class="surface-button-secondary px-4 py-3"
        :disabled="!canEvaluate || isEvaluating"
        @click="$emit('evaluate')"
      >
        {{ isEvaluating ? '评估中...' : '评估并入库' }}
      </button>
    </div>

    <div v-if="decision" class="mt-5 grid gap-4">
      <article
        class="rounded-3xl border px-4 py-4"
        :class="decision.accepted
          ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-950/30'
          : 'border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-900'"
      >
        <div class="flex flex-wrap items-center gap-3">
          <span
            class="surface-badge"
            :class="decision.accepted
              ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-200'
              : 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-200'"
          >
            {{ decision.accepted ? '已纳入样本库' : '未纳入样本库' }}
          </span>
          <span class="text-sm text-slate-600 dark:text-slate-300">
            分数 {{ decision.score.toFixed(2) }} / 阈值 {{ decision.threshold.toFixed(2) }}
          </span>
          <span class="text-sm text-slate-600 dark:text-slate-300">
            当前样本库 {{ decision.bankSize }} 条
          </span>
        </div>

        <p v-if="decision.sampleId" class="mt-3 text-sm text-slate-700 dark:text-slate-200">
          Sample ID: {{ decision.sampleId }}
        </p>
        <p v-if="decision.duplicateOf" class="mt-2 text-sm text-slate-700 dark:text-slate-200">
          Duplicate of: {{ decision.duplicateOf }}
        </p>
      </article>

      <div class="grid gap-2">
        <p
          v-for="reason in decision.reasons"
          :key="reason"
          class="rounded-2xl bg-slate-50 px-3 py-2 text-sm text-slate-700 dark:bg-slate-900 dark:text-slate-200"
        >
          {{ reason }}
        </p>
      </div>
    </div>

    <div
      v-else
      class="mt-5 rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    />
  </section>
</template>
