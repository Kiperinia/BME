<script setup lang="ts">
import type {
  ExemplarFeedbackMode,
  ExemplarFeedbackResult,
  ExemplarRetrievalCandidate,
  ExemplarRetrievalResult,
} from '@/types/workspace'

defineProps<{
  retrieval: ExemplarRetrievalResult | null
  feedbackMap: Record<string, ExemplarFeedbackResult>
  isRetrieving: boolean
  feedbackSubmittingFor: string | null
}>()

const emit = defineEmits<{
  (event: 'refresh'): void
  (event: 'feedback', exemplarId: string, mode: ExemplarFeedbackMode): void
}>()

const polarityTone: Record<ExemplarRetrievalCandidate['polarity'], string> = {
  positive: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-200',
  negative: 'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-200',
  boundary: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200',
}

const feedbackActions: Array<{ label: string; mode: ExemplarFeedbackMode }> = [
  { label: '检索有效', mode: 'success' },
  { label: '漏检样本', mode: 'false_negative' },
  { label: '误检样本', mode: 'false_positive' },
  { label: '存在不确定', mode: 'uncertain' },
]

const resolvePolarityLabel = (polarity: ExemplarRetrievalCandidate['polarity']) => {
  if (polarity === 'positive') return '正样本'
  if (polarity === 'negative') return '负样本'
  return '边界样本'
}
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex flex-col gap-4 border-b border-slate-200 pb-4 dark:border-slate-700 xl:flex-row xl:items-end xl:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">样本检索先验</h3>
        <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
          展示注入到 MedicalSAM3 的 exemplar 检索结果，并支持医生回写反馈。
        </p>
      </div>

      <button
        type="button"
        class="surface-button-secondary px-4 py-3"
        :disabled="isRetrieving"
        @click="$emit('refresh')"
      >
        {{ isRetrieving ? '刷新中...' : '刷新检索结果' }}
      </button>
    </div>

    <div v-if="retrieval" class="mt-5 grid gap-4">
      <div class="grid gap-3 md:grid-cols-4">
        <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
          <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">检索置信度</p>
          <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ retrieval.confidence.toFixed(2) }}</p>
        </article>
        <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
          <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">不确定性</p>
          <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ retrieval.uncertainty.toFixed(2) }}</p>
        </article>
        <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
          <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">候选样本</p>
          <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ retrieval.candidateCount }}</p>
        </article>
        <article class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
          <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">Prompt Token</p>
          <p class="mt-2 text-sm font-semibold text-slate-900 dark:text-white">{{ retrieval.promptTokenShape.join(' × ') || 'n/a' }}</p>
        </article>
      </div>

      <article class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="text-sm font-semibold text-slate-900 dark:text-white">先验注入键</p>
          <span class="surface-badge bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-200">
            {{ retrieval.bankId }}
          </span>
        </div>

        <div class="mt-3 flex flex-wrap gap-2">
          <span
            v-for="key in retrieval.priorKeys"
            :key="key"
            class="rounded-full bg-sky-100 px-3 py-1.5 text-sm font-medium text-sky-700 dark:bg-sky-900/50 dark:text-sky-200"
          >
            {{ key }}
          </span>
        </div>
      </article>

      <div class="grid gap-3">
        <article
          v-for="candidate in retrieval.candidates"
          :key="candidate.exemplarId"
          class="rounded-3xl border border-slate-200 p-4 dark:border-slate-700"
        >
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="flex flex-wrap items-center gap-2">
                <span class="surface-badge" :class="polarityTone[candidate.polarity]">
                  {{ resolvePolarityLabel(candidate.polarity) }}
                </span>
                <span class="text-sm font-semibold text-slate-900 dark:text-white">{{ candidate.exemplarId }}</span>
              </div>
              <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
                相似度 {{ candidate.similarity.toFixed(2) }} · 排序分 {{ candidate.rankScore.toFixed(2) }} · 惩罚项 {{ candidate.uncertaintyPenalty.toFixed(2) }}
              </p>
            </div>

            <div class="flex flex-wrap gap-2">
              <button
                v-for="action in feedbackActions"
                :key="action.mode"
                type="button"
                class="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-sky-300 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-700 dark:text-slate-200"
                :disabled="feedbackSubmittingFor === candidate.exemplarId"
                @click="$emit('feedback', candidate.exemplarId, action.mode)"
              >
                {{ feedbackSubmittingFor === candidate.exemplarId ? '提交中...' : action.label }}
              </button>
            </div>
          </div>

          <div v-if="candidate.tags.length" class="mt-3 flex flex-wrap gap-2">
            <span
              v-for="tag in candidate.tags"
              :key="tag"
              class="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200"
            >
              {{ tag }}
            </span>
          </div>

          <article
            v-if="feedbackMap[candidate.exemplarId]"
            class="mt-3 rounded-2xl bg-emerald-50 px-3 py-3 text-sm text-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-200"
          >
            <p class="font-semibold">最新状态：{{ feedbackMap[candidate.exemplarId]?.updatedState }}</p>
            <p class="mt-1">
              质量评分 {{ feedbackMap[candidate.exemplarId]?.qualityBreakdown.overall?.toFixed?.(2) ?? feedbackMap[candidate.exemplarId]?.qualityBreakdown.overall }}
            </p>
          </article>
        </article>
      </div>
    </div>

    <div
      v-else
      class="mt-5 rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    >
      暂无检索结果。完成分割后系统会自动拉取 exemplar 候选，也可以在调整专家配置后手动刷新。
    </div>
  </section>
</template>
