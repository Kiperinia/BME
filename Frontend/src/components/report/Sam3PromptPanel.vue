<script setup lang="ts">
const props = defineProps<{
  polypCount: number
  promptText: string
}>()

const emit = defineEmits<{
  (event: 'update:promptText', value: string): void
}>()

const handleInput = (event: Event) => {
  const target = event.target as HTMLTextAreaElement
  emit('update:promptText', target.value)
}
</script>

<template>
  <section class="surface-card flex h-full min-h-0 flex-col p-3.5">
    <div class="flex items-start justify-between gap-3 border-b border-gray-100 pb-3 dark:border-slate-700">
      <div>
        <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-100">SAM3 文本提示</h3>
        <p class="mt-1 text-xs leading-5 text-gray-500 dark:text-gray-400">
          为后续分割接口预留文本提示词入口，可辅助描述目标病灶或关注区域。
        </p>
      </div>

      <div class="rounded-xl bg-sky-50 px-3 py-2 text-right dark:bg-sky-950/60">
        <p class="text-[11px] uppercase tracking-[0.12em] text-sky-700 dark:text-sky-200">已发现息肉</p>
        <p class="mt-1 text-lg font-semibold text-sky-900 dark:text-white">{{ props.polypCount }}</p>
      </div>
    </div>

    <label class="mt-3 flex min-h-0 flex-1 flex-col gap-2">
      <span class="text-xs font-medium text-gray-600 dark:text-gray-300">SAM3 Prompt</span>
      <textarea
        class="surface-input min-h-0 flex-1 resize-none"
        :value="props.promptText"
        placeholder="例如：请优先分割镜头中心区域的隆起型息肉边界。"
        @input="handleInput"
      />
    </label>

    <p class="mt-3 text-xs leading-5 text-gray-500 dark:text-gray-400">
      当前为前端工作台输入区，后续可直接对接 SAM3 分割请求体中的文本提示字段。
    </p>
  </section>
</template>