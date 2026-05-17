<script setup lang="ts">
import type { WorkspacePatient } from '@/types/workspace'

const props = defineProps<{
  modelValue: WorkspacePatient
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: WorkspacePatient): void
}>()

const updateField = <K extends keyof WorkspacePatient>(field: K, value: WorkspacePatient[K]) => {
  emit('update:modelValue', {
    ...props.modelValue,
    [field]: value,
  })
}
</script>

<template>
  <section class="surface-card p-4">
    <div class="border-b border-slate-200 pb-3 dark:border-slate-700">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">患者信息</h3>
    </div>

    <div class="mt-3 grid gap-3 md:grid-cols-3">
      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">患者编号</span>
        <input
          :value="modelValue.patientId"
          class="surface-input mt-1.5"
          placeholder="如 PATIENT-20260517-001"
          @input="updateField('patientId', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">患者姓名</span>
        <input
          :value="modelValue.patientName"
          class="surface-input mt-1.5"
          placeholder="如 张某某"
          @input="updateField('patientName', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">检查日期</span>
        <input
          :value="modelValue.examDate"
          type="date"
          class="surface-input mt-1.5"
          @input="updateField('examDate', ($event.target as HTMLInputElement).value)"
        >
      </label>
    </div>
  </section>
</template>
