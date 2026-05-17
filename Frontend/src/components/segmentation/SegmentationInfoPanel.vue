<script setup lang="ts">
import { computed } from 'vue'

import type { UploadedWorkspaceImage, WorkspaceSegmentation } from '@/types/workspace'

const props = defineProps<{
  image: UploadedWorkspaceImage | null
  segmentation: WorkspaceSegmentation | null
  showMask: boolean
}>()

const infoItems = computed(() => {
  const image = props.image
  const segmentation = props.segmentation

  return [
    { label: '文件名', value: image?.filename || '未上传' },
    { label: '图像尺寸', value: image ? `${image.width} x ${image.height}` : '-' },
    { label: '边界框', value: segmentation ? segmentation.boundingBox.join(', ') : '-' },
    { label: '轮廓点数', value: segmentation ? String(segmentation.pointCount) : '-' },
    { label: '掩码面积', value: segmentation ? `${Math.round(segmentation.maskAreaPixels)} px²` : '-' },
    { label: '面积占比', value: segmentation ? `${(segmentation.maskAreaRatio * 100).toFixed(2)}%` : '-' },
    { label: '掩码状态', value: props.showMask ? '开启' : '关闭' },
  ]
})

const fileSizeLabel = computed(() => {
  if (!props.image) {
    return '-'
  }

  return `${(props.image.sizeBytes / 1024 / 1024).toFixed(2)} MB`
})
</script>

<template>
  <section class="surface-card p-5">
    <div class="border-b border-slate-200 pb-4 dark:border-slate-700">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">分割信息</h3>
    </div>

    <div class="mt-5 grid gap-3">
      <div class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
        <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">文件大小</p>
        <p class="mt-1 text-sm font-medium text-slate-900 dark:text-white">{{ fileSizeLabel }}</p>
      </div>

      <div
        v-for="item in infoItems"
        :key="item.label"
        class="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900"
      >
        <p class="text-xs uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">{{ item.label }}</p>
        <p class="mt-1 break-words text-sm font-medium text-slate-900 dark:text-white">{{ item.value }}</p>
      </div>
    </div>
  </section>
</template>
