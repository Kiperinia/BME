<script setup lang="ts">
defineProps<{
  enabled: boolean
  disabled?: boolean
}>()

defineEmits<{
  (event: 'toggle'): void
}>()
</script>

<template>
  <button
    type="button"
    class="inline-flex items-center gap-3 rounded-full border px-4 py-2 text-sm font-medium transition"
    :class="enabled
      ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200'
      : 'border-slate-200 bg-white text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300'"
    :disabled="disabled"
    @click="$emit('toggle')"
  >
    <span
      class="relative h-6 w-11 rounded-full transition"
      :class="enabled ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-slate-700'"
    >
      <span
        class="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition"
        :class="enabled ? 'left-[22px]' : 'left-0.5'"
      />
    </span>
    {{ enabled ? '掩码已开启' : '掩码已关闭' }}
  </button>
</template>
