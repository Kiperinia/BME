<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    visible: boolean
    message: string
    tone?: 'info' | 'success' | 'error'
  }>(),
  {
    tone: 'info',
  },
)

const toneClasses = computed(() => {
  if (props.tone === 'success') {
    return 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/70 dark:bg-emerald-950/70 dark:text-emerald-200'
  }

  if (props.tone === 'error') {
    return 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/70 dark:bg-rose-950/70 dark:text-rose-200'
  }

  return 'border-blue-200 bg-blue-50 text-blue-700 dark:border-sky-900/70 dark:bg-sky-950/70 dark:text-sky-200'
})
</script>

<template>
  <Transition
    enter-active-class="transition duration-200 ease-out"
    enter-from-class="translate-y-2 opacity-0"
    enter-to-class="translate-y-0 opacity-100"
    leave-active-class="transition duration-150 ease-in"
    leave-from-class="translate-y-0 opacity-100"
    leave-to-class="translate-y-2 opacity-0"
  >
    <div
      v-if="visible && message"
      class="fixed right-6 top-6 z-50 max-w-sm rounded-2xl border px-4 py-3 text-sm shadow-soft"
      :class="toneClasses"
      role="status"
      aria-live="polite"
    >
      {{ message }}
    </div>
  </Transition>
</template>