<script setup lang="ts">
import ParisClassificationBuilder from '@/components/workspace/ParisClassificationBuilder.vue'
import {
  formatParisClassification,
  type DetailedParisConfiguration,
  type ExpertConfiguration,
} from '@/types/workspace'

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

const updateParisDetail = (value: DetailedParisConfiguration) => {
  emit('update:modelValue', {
    ...props.modelValue,
    parisDetail: value,
    parisClassification: formatParisClassification(value),
  })
}
</script>

<template>
  <section class="surface-card p-5">
    <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">专家配置</h3>
    </div>

    <div class="mt-5 grid gap-4">
      <div>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Paris 分型细化</span>
        <ParisClassificationBuilder class="mt-2" :model-value="modelValue.parisDetail" @update:model-value="updateParisDetail" />
      </div>

      <label>
        <span class="text-xs font-medium uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">Paris 分型摘要</span>
        <textarea
          :value="modelValue.parisClassification"
          class="surface-input mt-2 min-h-[88px]"
          readonly
        />
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
