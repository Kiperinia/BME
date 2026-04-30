<script setup lang="ts">
import { computed, watch } from 'vue'

import type { AnnotationTag, FetchAnnotationTagsRequest, ReportContextData } from '@/types/eis'

const props = withDefaults(
  defineProps<{
    contextData: ReportContextData
    reportSnippet: string
    tags?: AnnotationTag[]
    isLoading?: boolean
    errorMessage?: string
  }>(),
  {
    tags: () => [],
    isLoading: false,
    errorMessage: '',
  },
)

const emit = defineEmits<{
  (event: 'fetch-agent-tags', payload: FetchAnnotationTagsRequest): void
  (event: 'tag-click', payload: AnnotationTag): void
}>()

const hasTags = computed(() => props.tags.length > 0)

watch(
  () => [props.contextData.videoFrameData.frameId, props.reportSnippet, props.contextData.captureImageSrcs.length],
  () => {
    if (!props.reportSnippet.trim()) {
      return
    }

    emit('fetch-agent-tags', {
      contextData: props.contextData,
      reportSnippet: props.reportSnippet,
    })
  },
  { immediate: true },
)
</script>

<template>
  <section class="surface-card flex h-full min-h-0 flex-col p-4">
    <div class="flex items-center justify-between gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">智能标签索引</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          Agent 依据视频帧与报告片段交叉生成定位标签。
        </p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-200">
        {{ props.contextData.videoFrameData.suspectedLocation }}
      </span>
    </div>

    <div v-if="props.isLoading" class="mt-4 flex flex-wrap gap-2">
      <div
        v-for="index in 6"
        :key="index"
        class="h-8 w-24 animate-pulse rounded-full bg-gray-100 dark:bg-slate-700"
      />
    </div>

    <div
      v-else-if="hasTags"
      class="mt-4 flex flex-wrap gap-2"
    >
      <button
        v-for="tag in props.tags"
        :key="tag.id"
        type="button"
        class="rounded-full border px-3 py-1.5 text-left text-xs transition hover:-translate-y-0.5 active:scale-[0.98] md:text-sm"
        :class="tag.needsReview
          ? 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-200'
          : 'border-blue-200 bg-blue-50 text-blue-700 dark:border-sky-800 dark:bg-sky-950/70 dark:text-sky-200'"
        @click="emit('tag-click', tag)"
      >
        <span class="font-medium">{{ tag.label }}</span>
        <span class="ml-2 text-xs opacity-80">{{ (tag.confidence * 100).toFixed(0) }}%</span>
      </button>
    </div>

    <p v-else class="mt-4 rounded-xl bg-gray-50 px-3 py-3 text-xs text-gray-500 dark:bg-slate-900 dark:text-gray-400 md:text-sm">
      当前尚无 Agent 标签输出，可调整报告片段或重新触发分析。
    </p>

    <p v-if="props.errorMessage" class="mt-3 text-xs text-rose-600 dark:text-rose-300 md:text-sm">
      {{ props.errorMessage }}
    </p>
  </section>
</template>