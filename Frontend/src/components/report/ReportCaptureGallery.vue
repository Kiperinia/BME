<script setup lang="ts">
defineProps<{
  images: string[]
}>()
</script>

<template>
  <div class="surface-card flex h-full min-h-0 flex-col overflow-hidden p-3">
    <div class="flex flex-col gap-1.5 border-b border-gray-100 pb-2.5 dark:border-slate-700 md:flex-row md:items-center md:justify-between">
      <div>
        <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-100">抓拍切片</h3>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">抓帧结果会直接沉淀在当前检查上下文中。</p>
      </div>
      <span class="surface-badge bg-slate-100 text-xs text-slate-600 dark:bg-slate-700 dark:text-slate-200">
        {{ images.length }} 张图像
      </span>
    </div>

    <div v-if="images.length" class="mt-2.5 grid min-h-0 flex-1 auto-rows-fr gap-2 sm:grid-cols-3 xl:grid-cols-3">
      <div
        v-for="(imageSrc, index) in images"
        :key="`${imageSrc}-${index}`"
        class="overflow-hidden rounded-xl border border-gray-200 bg-gray-50 dark:border-slate-700 dark:bg-slate-900"
      >
        <img :src="imageSrc" :alt="`抓拍图像 ${index + 1}`" class="aspect-[16/11] h-full w-full object-cover" />
      </div>
    </div>

    <div
      v-else
      class="mt-2.5 flex min-h-0 flex-1 items-center justify-center rounded-xl border border-dashed border-gray-200 bg-gray-50 px-4 text-center text-xs text-gray-500 dark:border-slate-700 dark:bg-slate-900 dark:text-gray-400 md:text-sm"
    >
      尚未抓取上下文图像，点击播放器中的“抓帧”后会显示在这里。
    </div>
  </div>
</template>