<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  imageUrl?: string
  imageWidth?: number
  imageHeight?: number
  maskCoordinates: [number, number][]
  showMask: boolean
}>()

const hasImage = computed(() => Boolean(props.imageUrl))
const hasMask = computed(() => props.maskCoordinates.length >= 3)
const polygonPoints = computed(() => props.maskCoordinates.map((point) => point.join(',')).join(' '))
const svgViewBox = computed(() => `0 0 ${props.imageWidth || 1} ${props.imageHeight || 1}`)
</script>

<template>
  <section class="surface-card p-5">
    <div class="flex items-center justify-between gap-3 border-b border-slate-200 pb-4 dark:border-slate-700">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">分割结果预览</h3>
        <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">原图、分割结果和掩码叠加视图同步展示。</p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
        {{ hasMask ? '已生成轮廓' : '等待分割' }}
      </span>
    </div>

    <div v-if="hasImage" class="mt-5 grid gap-4 xl:grid-cols-3">
      <article class="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 dark:border-slate-700">
        <div class="border-b border-slate-200 px-4 py-3 text-sm font-medium text-white/80 dark:border-slate-700">原图</div>
        <div class="relative aspect-[4/3]">
          <img :src="imageUrl" alt="原始图像" class="h-full w-full object-contain" />
        </div>
      </article>

      <article class="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 dark:border-slate-700">
        <div class="border-b border-slate-200 px-4 py-3 text-sm font-medium text-white/80 dark:border-slate-700">分割图</div>
        <div class="relative aspect-[4/3]">
          <img :src="imageUrl" alt="分割图像" class="h-full w-full object-contain opacity-30" />
          <svg
            v-if="hasMask"
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
        <div class="relative aspect-[4/3]">
          <img :src="imageUrl" alt="掩码叠加图像" class="h-full w-full object-contain" />
          <svg
            v-if="hasMask && showMask"
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
      class="mt-5 flex aspect-[16/7] items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-400"
    >
      上传图像后将在这里显示分割预览。
    </div>
  </section>
</template>
