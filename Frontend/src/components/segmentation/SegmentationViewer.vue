<script setup lang="ts">
import { computed, ref } from 'vue'

import MaskToggle from '@/components/segmentation/MaskToggle.vue'

const props = defineProps<{
  isSegmenting: boolean
  hasMaskImage: boolean
  imageUrl?: string
  imageWidth?: number
  imageHeight?: number
  maskDataUrl?: string
  maskCoordinates: [number, number][]
  showMask: boolean
  selectedFilename?: string
  selectedFileSizeLabel?: string
  selectedMaskFilename?: string
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const maskInputRef = ref<HTMLInputElement | null>(null)

const hasImage = computed(() => Boolean(props.imageUrl))
const maskImageSrc = computed(() => props.maskDataUrl || '')
const hasMask = computed(() => Boolean(maskImageSrc.value) || props.maskCoordinates.length >= 3)
const polygonPoints = computed(() => props.maskCoordinates.map((point) => point.join(',')).join(' '))
const svgViewBox = computed(() => `0 0 ${props.imageWidth || 1} ${props.imageHeight || 1}`)
const canToggleMask = computed(() => Boolean(maskImageSrc.value) || props.maskCoordinates.length > 0)

const emit = defineEmits<{
  (event: 'toggle-mask'): void
  (event: 'select-file', file: File): void
  (event: 'select-mask-file', file: File): void
  (event: 'segment'): void
  (event: 'apply-mask'): void
}>()

const pickFile = () => {
  inputRef.value?.click()
}

const pickMaskFile = () => {
  maskInputRef.value?.click()
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

const handleMaskFileSelection = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) {
    return
  }

  emit('select-mask-file', file)
  target.value = ''
}
</script>

<template>
  <section class="surface-card p-4">
    <input ref="inputRef" type="file" accept="image/*" class="hidden" @change="handleFileSelection">
    <input ref="maskInputRef" type="file" accept="image/*" class="hidden" @change="handleMaskFileSelection">

    <div class="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-700">
      <div>
        <p class="text-sm font-medium text-slate-900 dark:text-white">
          {{ selectedFilename || '尚未选择图像' }}
        </p>
        <p v-if="selectedFileSizeLabel" class="mt-1 text-sm text-slate-500 dark:text-slate-400">{{ selectedFileSizeLabel }}</p>
        <p v-if="selectedMaskFilename" class="mt-1 text-xs text-emerald-600 dark:text-emerald-300">
          已载入掩码图：{{ selectedMaskFilename }}
        </p>
      </div>

      <div class="flex flex-wrap gap-2">
        <button type="button" class="surface-button-secondary px-4 py-2.5 text-sm" @click="pickFile">
          选择本地图像
        </button>
        <button type="button" class="surface-button-secondary px-4 py-2.5 text-sm" :disabled="!hasImage" @click="pickMaskFile">
          上传已有掩码
        </button>
        <button
          type="button"
          class="surface-button-primary px-4 py-2.5 text-sm"
          :disabled="!hasImage || isSegmenting"
          @click="$emit('segment')"
        >
          {{ isSegmenting ? '分割中...' : '开始分割' }}
        </button>
        <button
          type="button"
          class="surface-button-primary px-4 py-2.5 text-sm"
          :disabled="!hasImage || !hasMaskImage"
          @click="$emit('apply-mask')"
        >
          应用掩码展示
        </button>
      </div>
    </div>

    <div class="flex items-center justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-700">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">分割结果预览</h3>
      </div>
      <div class="flex items-center gap-2">
        <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          {{ hasMask ? '已生成轮廓' : '等待分割' }}
        </span>
        <MaskToggle :enabled="showMask" :disabled="!canToggleMask" @toggle="$emit('toggle-mask')" />
      </div>
    </div>

    <div v-if="hasImage" class="mt-4 grid gap-3 xl:grid-cols-3">
      <article class="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 dark:border-slate-700">
        <div class="border-b border-slate-200 px-4 py-3 text-sm font-medium text-white/80 dark:border-slate-700">原图</div>
        <div class="relative aspect-[16/10]">
          <img :src="imageUrl" alt="原始图像" class="h-full w-full object-contain" />
        </div>
      </article>

      <article class="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 dark:border-slate-700">
        <div class="border-b border-slate-200 px-4 py-3 text-sm font-medium text-white/80 dark:border-slate-700">分割图</div>
        <div class="relative aspect-[16/10]">
          <img :src="imageUrl" alt="分割图像" class="h-full w-full object-contain opacity-30" />
          <img
            v-if="maskImageSrc"
            :src="maskImageSrc"
            alt="分割遮罩图"
            class="absolute inset-0 h-full w-full object-contain"
          />
          <svg
            v-else-if="hasMask"
            class="absolute inset-0 h-full w-full"
            :viewBox="svgViewBox"
            preserveAspectRatio="xMidYMid meet"
          >
            <polygon
              :points="polygonPoints"
              fill="rgba(16, 185, 129, 0.72)"
              stroke="rgba(5, 150, 105, 0.98)"
              stroke-width="5"
              stroke-linejoin="round"
            />
          </svg>
        </div>
      </article>

      <article class="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 dark:border-slate-700">
        <div class="border-b border-slate-200 px-4 py-3 text-sm font-medium text-white/80 dark:border-slate-700">掩码叠加</div>
        <div class="relative aspect-[16/10]">
          <img :src="imageUrl" alt="掩码叠加图像" class="h-full w-full object-contain" />
          <img
            v-if="maskImageSrc && showMask"
            :src="maskImageSrc"
            alt="掩码叠加图层"
            class="absolute inset-0 h-full w-full object-contain"
          />
          <svg
            v-else-if="hasMask && showMask"
            class="absolute inset-0 h-full w-full"
            :viewBox="svgViewBox"
            preserveAspectRatio="xMidYMid meet"
          >
            <polygon
              :points="polygonPoints"
              fill="rgba(59, 130, 246, 0.24)"
              stroke="rgba(56, 189, 248, 0.98)"
              stroke-width="5"
              stroke-linejoin="round"
            />
          </svg>
        </div>
      </article>
    </div>

    <div
      v-else
      class="mt-4 flex aspect-[16/5] items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    />
  </section>
</template>
