<script setup lang="ts">
import { computed, watch } from 'vue'

import type { AnnotationTag, FetchAnnotationTagsRequest, VideoFrameData } from '@/types/eis'

const props = withDefaults(
  defineProps<{
    videoFrameData: VideoFrameData
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
  () => [props.videoFrameData.frameId, props.reportSnippet],
  () => {
    if (!props.reportSnippet.trim()) {
      return
    }

    emit('fetch-agent-tags', {
      videoFrameData: props.videoFrameData,
      reportSnippet: props.reportSnippet,
    })
  },
  { immediate: true },
)
</script>

<template>
  <section class="surface-card p-6">
    <div class="flex items-center justify-between gap-4">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-gray-100">智能标签索引</h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Agent 依据视频帧与报告片段交叉生成定位标签。
        </p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-200">
        {{ props.videoFrameData.suspectedLocation }}
      </span>
    </div>

    <div v-if="props.isLoading" class="mt-6 flex flex-wrap gap-3">
      <div
        v-for="index in 6"
        :key="index"
        class="h-10 w-28 animate-pulse rounded-full bg-gray-100 dark:bg-slate-700"
      />
    </div>

    <div
      v-else-if="hasTags"
      class="mt-6 flex flex-wrap gap-3"
    >
      <button
        v-for="tag in props.tags"
        :key="tag.id"
        type="button"
        class="rounded-full border px-4 py-2 text-left text-sm transition hover:-translate-y-0.5 active:scale-[0.98]"
        :class="tag.needsReview
          ? 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-200'
          : 'border-blue-200 bg-blue-50 text-blue-700 dark:border-sky-800 dark:bg-sky-950/70 dark:text-sky-200'"
        @click="emit('tag-click', tag)"
      >
        <span class="font-medium">{{ tag.label }}</span>
        <span class="ml-2 text-xs opacity-80">{{ (tag.confidence * 100).toFixed(0) }}%</span>
      </button>
    </div>

    <p v-else class="mt-6 rounded-xl bg-gray-50 px-4 py-4 text-sm text-gray-500 dark:bg-slate-900 dark:text-gray-400">
      当前尚无 Agent 标签输出，可调整报告片段或重新触发分析。
    </p>

    <p v-if="props.errorMessage" class="mt-4 text-sm text-rose-600 dark:text-rose-300">
      {{ props.errorMessage }}
    </p>
  </section>
</template>