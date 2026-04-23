<script setup lang="ts">
defineProps<{
  images: string[]
}>()
</script>

<template>
  <div class="surface-card p-6">
    <div class="flex flex-col gap-3 border-b border-gray-100 pb-4 dark:border-slate-700 md:flex-row md:items-center md:justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-800 dark:text-gray-100">抓拍上下文</h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">抓帧结果会被直接纳入 Agent 上下文。</p>
      </div>
      <span class="surface-badge bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-200">
        {{ images.length }} 张图像
      </span>
    </div>

    <div class="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      <div
        v-for="(imageSrc, index) in images"
        :key="`${imageSrc}-${index}`"
        class="overflow-hidden rounded-2xl border border-gray-200 bg-gray-50 dark:border-slate-700 dark:bg-slate-900"
      >
        <img :src="imageSrc" :alt="`抓拍图像 ${index + 1}`" class="aspect-[4/3] h-full w-full object-cover" />
      </div>
    </div>
  </div>
</template>