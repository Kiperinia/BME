<script setup lang="ts">
import type { ExpertConfiguration } from '@/types/workspace'

const props = defineProps<{
  modelValue: ExpertConfiguration
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: ExpertConfiguration): void
}>()

const updateField = <K extends keyof ExpertConfiguration>(field: K, value: ExpertConfiguration[K]) => {
  emit('update:modelValue', {
    ...props.modelValue,
    [field]: value,
  })
}
</script>

<template>
  <section class="surface-card p-5">
    <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">专家配置</h3>
      <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">医生可在此补充 Paris 分型、病理/类型分类与备注信息。</p>
    </div>

    <div class="mt-5 grid gap-4">
      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Paris 分型</span>
        <input
          :value="modelValue.parisClassification"
          class="surface-input mt-2"
          placeholder="如 Paris 0-Is"
          @input="updateField('parisClassification', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">类型分类</span>
        <input
          :value="modelValue.lesionType"
          class="surface-input mt-2"
          placeholder="如 腺瘤样息肉 / 早癌可疑"
          @input="updateField('lesionType', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">病理/类型分类</span>
        <input
          :value="modelValue.pathologyClassification"
          class="surface-input mt-2"
          placeholder="如 管状腺瘤，待病理证实"
          @input="updateField('pathologyClassification', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">表面模式</span>
        <input
          :value="modelValue.surfacePattern"
          class="surface-input mt-2"
          placeholder="如 表面颗粒样，血管纹理不规则"
          @input="updateField('surfacePattern', ($event.target as HTMLInputElement).value)"
        >
      </label>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">备注</span>
        <textarea
          :value="modelValue.expertNotes"
          class="surface-input mt-2 min-h-[120px]"
          placeholder="补充人工判断、病灶背景、复核建议等。"
          @input="updateField('expertNotes', ($event.target as HTMLTextAreaElement).value)"
        />
      </label>
    </div>
  </section>
</template>
