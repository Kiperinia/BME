<script setup lang="ts">
import type { ThemeMode } from '@/stores/theme'

defineProps<{
  isDark: boolean
  mode: ThemeMode
}>()

const emit = defineEmits<{
  (event: 'toggle'): void
}>()
</script>

<template>
  <button
    type="button"
    class="surface-button-secondary gap-3 px-4 py-2.5"
    :aria-label="isDark ? '切换到浅色模式' : '切换到深色模式'"
    @click="emit('toggle')"
  >
    <span
      class="inline-flex h-9 w-9 items-center justify-center rounded-full bg-gray-100 text-gray-700 dark:bg-slate-800 dark:text-sky-300"
    >
      <svg
        v-if="isDark"
        class="h-5 w-5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"
        />
      </svg>
      <svg
        v-else
        class="h-5 w-5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
      >
        <circle cx="12" cy="12" r="4" />
        <path stroke-linecap="round" d="M12 2v2.5M12 19.5V22M4.93 4.93l1.77 1.77M17.3 17.3l1.77 1.77M2 12h2.5M19.5 12H22M4.93 19.07l1.77-1.77M17.3 6.7l1.77-1.77" />
      </svg>
    </span>
    <span class="flex flex-col items-start text-left">
      <span class="text-sm font-medium text-gray-700 dark:text-gray-100">
        {{ isDark ? '深色模式' : '浅色模式' }}
      </span>
      <span class="text-xs text-gray-500 dark:text-gray-400">
        {{ mode === 'system' ? '当前跟随系统' : '用户手动选择' }}
      </span>
    </span>
  </button>
</template>