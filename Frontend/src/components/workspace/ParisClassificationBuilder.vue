<script setup lang="ts">
import {
  PARIS_GROUP_LABELS,
  PARIS_SUBTYPE_OPTIONS,
  createParisDetailFromSelection,
  formatParisClassification,
  type DetailedParisConfiguration,
  type ParisMorphologyGroup,
} from '@/types/workspace'

const props = defineProps<{
  modelValue: DetailedParisConfiguration
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: DetailedParisConfiguration): void
}>()

const updateGroup = (group: ParisMorphologyGroup) => {
  emit('update:modelValue', createParisDetailFromSelection(group, 0))
}

const updateSlider = (event: Event) => {
  const nextIndex = Number((event.target as HTMLInputElement).value)
  emit('update:modelValue', createParisDetailFromSelection(props.modelValue.morphologyGroup, nextIndex))
}
</script>

<template>
  <div class="grid gap-4 rounded-3xl border border-slate-200 p-4 dark:border-slate-700">
    <div class="flex flex-wrap gap-2">
      <button
        v-for="(label, group) in PARIS_GROUP_LABELS"
        :key="group"
        type="button"
        class="rounded-full px-4 py-2 text-sm font-medium transition"
        :class="modelValue.morphologyGroup === group
          ? 'bg-sky-600 text-white shadow-soft dark:bg-sky-500'
          : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'"
        @click="updateGroup(group as ParisMorphologyGroup)"
      >
        {{ label }}
      </button>
    </div>

    <div class="rounded-3xl bg-slate-50 p-4 dark:bg-slate-900">
      <div class="flex items-center justify-between gap-4">
        <div>
          <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">当前 Paris 选项</p>
          <p class="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
            {{ modelValue.subtypeCode }} · {{ modelValue.displayLabel }}
          </p>
        </div>
        <span class="surface-badge bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-200">
          {{ modelValue.selectedSubtypeIndex + 1 }} / {{ PARIS_SUBTYPE_OPTIONS[modelValue.morphologyGroup].length }}
        </span>
      </div>

      <input
        class="mt-4 w-full accent-sky-600"
        type="range"
        min="0"
        :max="PARIS_SUBTYPE_OPTIONS[modelValue.morphologyGroup].length - 1"
        :value="modelValue.selectedSubtypeIndex"
        @input="updateSlider"
      >

      <p class="mt-3 text-sm font-medium text-slate-800 dark:text-slate-100">{{ modelValue.featureSummary }}</p>
      <p class="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">{{ modelValue.featureReference }}</p>
      <p class="mt-3 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200">
        {{ formatParisClassification(modelValue) }}
      </p>
    </div>
  </div>
</template>
