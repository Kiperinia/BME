<script setup lang="ts">
defineProps<{
  savedAtLabel: string
  annotationCount: number
}>()
</script>

<template>
  <div class="surface-card p-4">
    <div class="flex items-center justify-between gap-4">
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">草稿状态</p>
        <p class="mt-1 text-base font-semibold text-gray-800 dark:text-gray-100">{{ savedAtLabel }}</p>
      </div>
      <span class="surface-badge bg-blue-50 text-xs text-blue-700 dark:bg-sky-950/70 dark:text-sky-200">
        {{ annotationCount }} 个标签
      </span>
    </div>
  </div>
</template>