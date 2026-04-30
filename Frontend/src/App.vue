<script setup lang="ts">
import { onMounted } from 'vue'
import { RouterLink, RouterView } from 'vue-router'

import ThemeToggleButton from '@/components/common/ThemeToggleButton.vue'
import { useThemeStore } from '@/stores/theme'

const themeStore = useThemeStore()

onMounted(async () => {
  themeStore.initializeTheme()
})
</script>

<template>
  <div class="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.12),_transparent_34%),linear-gradient(180deg,_#f8fafc_0%,_#eef2ff_100%)] dark:bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.16),_transparent_28%),linear-gradient(180deg,_#020617_0%,_#0f172a_100%)]">
    <header class="sticky top-0 z-40 border-b border-white/60 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/75">
      <div class="mx-auto flex max-w-[1800px] items-center justify-between gap-4 px-6 py-4 lg:px-8">
        <div>
          <p class="text-xs uppercase tracking-[0.2em] text-sky-600 dark:text-sky-300">BME Frontend</p>
          <h1 class="mt-1 text-xl font-semibold text-slate-900 dark:text-white">前端页面导航</h1>
        </div>

        <nav class="flex items-center gap-2 rounded-2xl bg-slate-100 p-1 dark:bg-slate-900">
          <RouterLink class="rounded-xl px-4 py-2 text-sm font-medium text-slate-600 transition hover:text-slate-900 dark:text-slate-300 dark:hover:text-white" active-class="bg-white text-slate-900 shadow-soft dark:bg-slate-800 dark:text-white" to="/report-builder">
            检查工作台
          </RouterLink>
          <RouterLink class="rounded-xl px-4 py-2 text-sm font-medium text-slate-600 transition hover:text-slate-900 dark:text-slate-300 dark:hover:text-white" active-class="bg-white text-slate-900 shadow-soft dark:bg-slate-800 dark:text-white" to="/report-generation">
            报告生成
          </RouterLink>
          <RouterLink class="rounded-xl px-4 py-2 text-sm font-medium text-slate-600 transition hover:text-slate-900 dark:text-slate-300 dark:hover:text-white" active-class="bg-white text-slate-900 shadow-soft dark:bg-slate-800 dark:text-white" to="/settings">
            系统设置
          </RouterLink>
        </nav>

        <ThemeToggleButton :is-dark="themeStore.isDark" :mode="themeStore.mode" @toggle="themeStore.toggleTheme" />
      </div>
    </header>

    <RouterView />
  </div>
</template>
