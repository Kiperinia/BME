<script setup lang="ts">
import { computed } from 'vue'

import type { PatientExamStatus, PatientGender } from '@/types/eis'

const props = defineProps<{
  patientName: string
  gender: PatientGender
  age: number
  patientId: string
  examDate: string
  status: PatientExamStatus
}>()

const emit = defineEmits<{
  (event: 'edit', patientId: string): void
  (event: 'view-history'): void
}>()

const statusMeta = computed(() => {
  if (props.status === 2) {
    return {
      label: '已出报告',
      className: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/70 dark:text-emerald-200',
    }
  }

  if (props.status === 1) {
    return {
      label: '检查中',
      className: 'bg-blue-100 text-blue-700 dark:bg-sky-950/70 dark:text-sky-200',
    }
  }

  return {
    label: '候诊',
    className: 'bg-gray-100 text-gray-600 dark:bg-slate-700 dark:text-slate-200',
  }
})

const infoItems = computed(() => [
  { label: '患者姓名', value: props.patientName },
  { label: '性别', value: props.gender },
  { label: '年龄', value: `${props.age} 岁` },
  { label: '病历号', value: props.patientId },
  { label: '检查日期', value: props.examDate },
])
</script>

<template>
  <article class="surface-card flex h-full min-h-0 flex-col p-3.5">
    <div class="flex flex-col gap-2.5 border-b border-gray-100 pb-2.5 dark:border-slate-700">
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">EIS 患者概览</p>
        <div class="mt-1.5 flex items-start justify-between gap-2.5">
          <div>
            <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">{{ patientName }}</h3>
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">病历号 {{ patientId }}</p>
          </div>
          <span class="surface-badge" :class="statusMeta.className">{{ statusMeta.label }}</span>
        </div>
      </div>
    </div>

    <div class="mt-3 grid flex-1 auto-rows-fr grid-cols-2 gap-2.5">
      <div
        v-for="item in infoItems"
        :key="item.label"
        class="rounded-xl bg-gray-50 px-2.5 py-2 dark:bg-slate-900"
      >
        <p class="text-xs uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">{{ item.label }}</p>
        <p class="mt-1 text-sm font-medium leading-5 text-gray-800 dark:text-gray-100">{{ item.value }}</p>
      </div>
    </div>

    <div class="mt-3 grid grid-cols-2 gap-2">
      <button type="button" class="surface-button-secondary px-3 py-1.5 text-sm" @click="emit('edit', patientId)">
        编辑
      </button>
      <button type="button" class="surface-button-secondary px-3 py-1.5 text-sm" @click="emit('view-history')">
        历史记录
      </button>
    </div>
  </article>
</template>