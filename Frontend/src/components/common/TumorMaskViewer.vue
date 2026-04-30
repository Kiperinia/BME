<script setup lang="ts">
import { computed, ref } from 'vue'

import type { PolygonMask, TumorDetails, TumorMaskData } from '@/types/eis'

const props = defineProps<{
  tumorImageSrc: string
  maskData: TumorMaskData
  details: TumorDetails
}>()

const emit = defineEmits<{
  (event: 'toggle-mask', visible: boolean): void
  (event: 'expand-view'): void
}>()

const imageWidth = ref(1200)
const imageHeight = ref(900)
const isMaskVisible = ref(true)
const isMagnifierVisible = ref(false)
const pointerPosition = ref({ x: 50, y: 50, ratioX: 0.5, ratioY: 0.5 })

const maskPolygons = computed<PolygonMask[]>(() => (Array.isArray(props.maskData) ? props.maskData : []))
const maskImageSrc = computed(() => (typeof props.maskData === 'string' ? props.maskData : ''))

const magnifierTransformStyle = computed(() => ({
  transform: 'scale(2.1)',
  transformOrigin: `${pointerPosition.value.ratioX * 100}% ${pointerPosition.value.ratioY * 100}%`,
}))

const detailItems = computed(() => [
  { label: '预估尺寸', value: `${props.details.estimatedSizeMm.toFixed(1)} mm` },
  { label: '分类', value: props.details.classification },
  { label: '位置', value: props.details.location },
  { label: '表面形态', value: props.details.surfacePattern },
  { label: '置信度', value: `${(props.details.confidence * 100).toFixed(0)}%` },
])

const handleImageLoad = (event: Event) => {
  const image = event.target as HTMLImageElement
  imageWidth.value = image.naturalWidth || 1200
  imageHeight.value = image.naturalHeight || 900
}

const handlePointerMove = (event: MouseEvent) => {
  const target = event.currentTarget as HTMLElement
  const rect = target.getBoundingClientRect()
  const ratioX = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1)
  const ratioY = Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1)

  pointerPosition.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    ratioX,
    ratioY,
  }
}

const toggleMaskVisibility = () => {
  isMaskVisible.value = !isMaskVisible.value
  emit('toggle-mask', isMaskVisible.value)
}
</script>

<template>
  <section class="surface-card flex h-full min-h-0 flex-col overflow-hidden">
    <div
      class="relative aspect-[16/10] cursor-zoom-in overflow-hidden bg-slate-950"
      @click="emit('expand-view')"
      @mouseenter="isMagnifierVisible = true"
      @mouseleave="isMagnifierVisible = false"
      @mousemove="handlePointerMove"
    >
      <img
        :src="tumorImageSrc"
        alt="肿瘤区域截图"
        class="h-full w-full object-cover"
        @load="handleImageLoad"
      />

      <svg
        v-if="maskPolygons.length && isMaskVisible"
        class="pointer-events-none absolute inset-0 h-full w-full"
        :viewBox="`0 0 ${imageWidth} ${imageHeight}`"
      >
        <polygon
          v-for="polygon in maskPolygons"
          :key="polygon.id"
          :points="polygon.points.map((point) => point.join(',')).join(' ')"
          :fill="polygon.fillColor ?? 'rgba(16, 185, 129, 0.26)'"
          :stroke="polygon.strokeColor ?? 'rgba(5, 150, 105, 0.95)'"
          stroke-width="6"
          stroke-linejoin="round"
        />
      </svg>

      <img
        v-else-if="maskImageSrc && isMaskVisible"
        :src="maskImageSrc"
        alt="Mask 图层"
        class="pointer-events-none absolute inset-0 h-full w-full object-cover"
      />

      <div class="absolute bottom-3 left-3 rounded-full bg-slate-950/70 px-2.5 py-1 text-[11px] text-white">
        点击图像查看放大细节
      </div>

      <div
        v-if="isMagnifierVisible"
        class="pointer-events-none absolute hidden h-28 w-28 overflow-hidden rounded-full border-4 border-white/80 shadow-soft md:block"
        :style="{
          left: `${pointerPosition.x}px`,
          top: `${pointerPosition.y}px`,
          transform: 'translate(-50%, -50%)',
        }"
      >
        <div class="absolute inset-0" :style="magnifierTransformStyle">
          <img :src="tumorImageSrc" alt="放大预览" class="h-full w-full object-cover" />
          <svg
            v-if="maskPolygons.length && isMaskVisible"
            class="absolute inset-0 h-full w-full"
            :viewBox="`0 0 ${imageWidth} ${imageHeight}`"
          >
            <polygon
              v-for="polygon in maskPolygons"
              :key="`${polygon.id}-zoom`"
              :points="polygon.points.map((point) => point.join(',')).join(' ')"
              :fill="polygon.fillColor ?? 'rgba(16, 185, 129, 0.26)'"
              :stroke="polygon.strokeColor ?? 'rgba(5, 150, 105, 0.95)'"
              stroke-width="6"
              stroke-linejoin="round"
            />
          </svg>
          <img
            v-else-if="maskImageSrc && isMaskVisible"
            :src="maskImageSrc"
            alt="放大遮罩预览"
            class="absolute inset-0 h-full w-full object-cover"
          />
        </div>
      </div>
    </div>

    <div class="flex min-h-0 flex-1 flex-col p-4">
      <div class="flex flex-col gap-3 border-b border-gray-100 pb-3 dark:border-slate-700 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h3 class="text-base font-semibold text-gray-800 dark:text-gray-100">肿瘤区域确认</h3>
          <p class="mt-1 text-xs text-gray-500 dark:text-gray-400 md:text-sm">用于局部 ROI 级别的病灶边界与形态核查。</p>
        </div>

        <button type="button" class="surface-button-secondary px-3 py-1.5 text-sm" @click.stop="toggleMaskVisibility">
          {{ isMaskVisible ? '隐藏遮罩' : '显示遮罩' }}
        </button>
      </div>

      <div class="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <div
          v-for="item in detailItems"
          :key="item.label"
          class="rounded-xl bg-gray-50 px-3 py-2.5 dark:bg-slate-900"
        >
          <p class="text-xs uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">{{ item.label }}</p>
          <p class="mt-1 text-sm font-medium text-gray-800 dark:text-gray-100">{{ item.value }}</p>
        </div>
      </div>
    </div>
  </section>
</template>