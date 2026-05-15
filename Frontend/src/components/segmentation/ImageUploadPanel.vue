<script setup lang="ts">
import { computed, ref } from 'vue'

const props = defineProps<{
  isSegmenting: boolean
  hasImage: boolean
  selectedFilename?: string
  selectedFileSizeLabel?: string
}>()

const emit = defineEmits<{
  (event: 'select-file', file: File): void
  (event: 'segment'): void
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)

const panelClass = computed(() => {
  if (isDragging.value) {
    return 'border-sky-400 bg-sky-50/80 dark:border-sky-500 dark:bg-sky-950/30'
  }

  return 'border-slate-200 bg-white/80 dark:border-slate-700 dark:bg-slate-900/70'
})

const pickFile = () => {
  inputRef.value?.click()
}

const handleFileSelection = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) {
    return
  }

  emit('select-file', file)
  target.value = ''
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    emit('select-file', file)
  }
}
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div class="max-w-3xl">
        <p class="text-xs uppercase tracking-[0.18em] text-sky-600 dark:text-sky-300">Workspace</p>
        <h2 class="mt-2 text-2xl font-semibold text-slate-900 dark:text-white">本地图像上传与 MedicalSAM3 分割</h2>
        <p class="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
          上传一帧内窥镜图像后，工作台会调用后端 SAM3 Runtime，生成轮廓、边界框和后续 Agent 报告所需的结构化输入。
        </p>
      </div>

      <div class="flex flex-wrap gap-3">
        <button type="button" class="surface-button-secondary px-4 py-3" @click="pickFile">
          选择本地图像
        </button>
        <button
          type="button"
          class="surface-button-primary px-4 py-3"
          :disabled="!hasImage || isSegmenting"
          @click="$emit('segment')"
        >
          {{ isSegmenting ? '分割中...' : '开始分割' }}
        </button>
      </div>
    </div>

    <div
      class="mt-5 rounded-3xl border border-dashed p-6 transition"
      :class="panelClass"
      @dragenter.prevent="isDragging = true"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop="handleDrop"
    >
      <input ref="inputRef" type="file" accept="image/*" class="hidden" @change="handleFileSelection">

      <div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <p class="text-sm font-medium text-slate-900 dark:text-white">
            {{ selectedFilename || '支持拖拽或点击选择 JPG / PNG / TIFF 图像' }}
          </p>
          <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {{ selectedFileSizeLabel || '建议上传单帧病灶图像，便于分割和专家复核。' }}
          </p>
        </div>

        <button type="button" class="surface-button-secondary px-4 py-2.5 text-sm" @click="pickFile">
          重新选择
        </button>
      </div>
    </div>
  </section>
</template>
