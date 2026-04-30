<script setup lang="ts">
defineProps<{
  initialOpinion: string
  findings: string
  conclusion: string
  layoutSuggestion: string
  streamText: string
  isAgentLoading: boolean
}>()

const emit = defineEmits<{
  (event: 'update:initialOpinion', value: string): void
  (event: 'update:findings', value: string): void
  (event: 'update:conclusion', value: string): void
  (event: 'update:layoutSuggestion', value: string): void
}>()

const updateField = (event: Event, field: 'initialOpinion' | 'findings' | 'conclusion' | 'layoutSuggestion') => {
  const target = event.target as HTMLTextAreaElement

  if (field === 'initialOpinion') {
    emit('update:initialOpinion', target.value)
    return
  }

  if (field === 'findings') {
    emit('update:findings', target.value)
    return
  }

  if (field === 'conclusion') {
    emit('update:conclusion', target.value)
    return
  }

  emit('update:layoutSuggestion', target.value)
}
</script>

<template>
  <div class="surface-card h-full min-h-0 p-4">
    <div class="grid h-full min-h-0 gap-4 xl:grid-cols-2">
      <div>
        <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">Agent 交互面板</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">
          左侧编辑医生初步意见，生成后继续修订正式报告字段。
        </p>
      </div>

      <div class="rounded-2xl bg-slate-950 p-3 text-sm text-slate-100 xl:row-span-2">
        <div class="flex items-center justify-between gap-3 border-b border-slate-800 pb-2.5">
          <span class="font-medium">流式输出</span>
          <span class="text-xs text-slate-400">{{ isAgentLoading ? 'Streaming' : 'Idle' }}</span>
        </div>
        <pre class="mt-3 min-h-24 whitespace-pre-wrap font-sans text-xs leading-6 text-slate-200 md:text-sm">{{ streamText || '等待 Agent 输出...' }}</pre>
      </div>

      <label class="block space-y-1.5">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-200">医生初步意见</span>
        <textarea
          class="surface-input min-h-20 text-sm"
          :value="initialOpinion"
          placeholder="补充病灶部位、形态、处理建议等重点描述。"
          @input="updateField($event, 'initialOpinion')"
        />
      </label>

      <label class="block space-y-1.5">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-200">检查所见</span>
        <textarea
          class="surface-input min-h-24 text-sm"
          :value="findings"
          placeholder="用于填写镜下所见、部位和形态特征。"
          @input="updateField($event, 'findings')"
        />
      </label>

      <label class="block space-y-1.5">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-200">诊断结论</span>
        <textarea
          class="surface-input min-h-20 text-sm"
          :value="conclusion"
          placeholder="用于填写诊断结论和建议。"
          @input="updateField($event, 'conclusion')"
        />
      </label>

      <label class="block space-y-1.5">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-200">排版建议</span>
        <textarea
          class="surface-input min-h-20 text-sm"
          :value="layoutSuggestion"
          placeholder="用于填写排版顺序、独立段落和书写注意事项。"
          @input="updateField($event, 'layoutSuggestion')"
        />
      </label>
    </div>
  </div>
</template>