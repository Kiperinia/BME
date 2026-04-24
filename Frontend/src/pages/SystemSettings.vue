<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import ThemeToggleButton from '@/components/common/ThemeToggleButton.vue'
import { useThemeStore, type ThemeMode } from '@/stores/theme'

type SectionId = 'general' | 'appearance' | 'viewer' | 'notifications' | 'ai' | 'dicom'

const themeStore = useThemeStore()

const sectionItems: Array<{ id: SectionId; label: string; description: string }> = [
  { id: 'general', label: '常规设置', description: '语言、自动保存与默认检查类型' },
  { id: 'appearance', label: '外观主题', description: '主题模式与界面偏好' },
  { id: 'viewer', label: '影像查看器', description: '布局、显示和测量习惯' },
  { id: 'notifications', label: '通知', description: '危急值、审核和提示音' },
  { id: 'ai', label: 'AI 辅助', description: '自动分析与置信度阈值' },
  { id: 'dicom', label: 'DICOM 连接', description: 'PACS 服务连接参数' },
]

const activeSection = ref<SectionId>('general')

const language = ref('zh-CN')
const autoSaveInterval = ref(30)
const defaultModality = ref('CT')
const imageQuality = ref('high')
const fontSize = ref('medium')

const notifyCritical = ref(true)
const notifyNewCase = ref(true)
const notifyReport = ref(true)
const soundEnabled = ref(false)

const defaultLayout = ref('2x2')
const showOverlay = ref(true)
const scrollDirection = ref('natural')
const crosshair = ref(false)
const measureUnit = ref('mm')

const aiAutoAnalyze = ref(true)
const aiConfidenceThreshold = ref(75)
const aiHighlightFindings = ref(true)

const dicomServerHost = ref('192.168.1.100')
const dicomServerPort = ref('4242')
const dicomAeTitle = ref('MEDIMAGEDX')
const dicomAutoFetch = ref(true)

const saved = ref(false)

const languageOptions = [
  { value: 'zh-CN', label: '简体中文' },
  { value: 'en-US', label: 'English' },
]

const modalityOptions = [
  { value: 'CT', label: 'CT' },
  { value: 'MRI', label: 'MRI' },
  { value: 'X-Ray', label: 'X-Ray' },
  { value: 'Ultrasound', label: '超声' },
  { value: 'PET', label: 'PET' },
]

const fontSizeOptions = [
  { value: 'small', label: '小 (14px)' },
  { value: 'medium', label: '中 (16px)' },
  { value: 'large', label: '大 (18px)' },
]

const layoutOptions = [
  { value: '1x1', label: '1×1 单窗格' },
  { value: '2x2', label: '2×2 四窗格' },
  { value: '1+2', label: '1+2 混合' },
]

const imageQualityOptions = [
  { value: 'low', label: '低质量（快速加载）' },
  { value: 'medium', label: '中等质量' },
  { value: 'high', label: '高质量' },
]

const scrollDirectionOptions = [
  { value: 'natural', label: '自然方向' },
  { value: 'reverse', label: '反向' },
]

const measureUnitOptions = [
  { value: 'mm', label: '毫米 (mm)' },
  { value: 'cm', label: '厘米 (cm)' },
]

const themeOptions: Array<{ value: ThemeMode; label: string; description: string }> = [
  { value: 'light', label: '浅色模式', description: '适合明亮环境和打印预览' },
  { value: 'dark', label: '深色模式', description: '降低夜间工作时的视觉刺激' },
  { value: 'system', label: '跟随系统', description: '自动同步操作系统主题' },
]

let saveTimer: number | undefined

const quickSummary = computed(() => [
  `${themeStore.mode === 'system' ? '系统' : '手动'}主题`,
  `${autoSaveInterval.value} 秒自动保存`,
  `AI 阈值 ${aiConfidenceThreshold.value}%`,
])

const setThemeMode = (mode: ThemeMode) => {
  if (mode === 'system') {
    themeStore.resetToSystem()
    return
  }

  themeStore.setMode(mode)
}

const handleSave = () => {
  saved.value = true

  if (saveTimer) {
    window.clearTimeout(saveTimer)
  }

  saveTimer = window.setTimeout(() => {
    saved.value = false
  }, 2200)
}

const resetDefaults = () => {
  language.value = 'zh-CN'
  autoSaveInterval.value = 30
  defaultModality.value = 'CT'
  imageQuality.value = 'high'
  fontSize.value = 'medium'
  notifyCritical.value = true
  notifyNewCase.value = true
  notifyReport.value = true
  soundEnabled.value = false
  defaultLayout.value = '2x2'
  showOverlay.value = true
  scrollDirection.value = 'natural'
  crosshair.value = false
  measureUnit.value = 'mm'
  aiAutoAnalyze.value = true
  aiConfidenceThreshold.value = 75
  aiHighlightFindings.value = true
  dicomServerHost.value = '192.168.1.100'
  dicomServerPort.value = '4242'
  dicomAeTitle.value = 'MEDIMAGEDX'
  dicomAutoFetch.value = true
  themeStore.resetToSystem()
  handleSave()
}

const updateActiveSection = () => {
  if (typeof window === 'undefined') {
    return
  }

  const offset = 180

  for (const item of [...sectionItems].reverse()) {
    const element = document.getElementById(item.id)

    if (element && element.getBoundingClientRect().top <= offset) {
      activeSection.value = item.id
      return
    }
  }

  activeSection.value = 'general'
}

onMounted(() => {
  updateActiveSection()
  window.addEventListener('scroll', updateActiveSection, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', updateActiveSection)

  if (saveTimer) {
    window.clearTimeout(saveTimer)
  }
})
</script>

<template>
  <main class="mx-auto w-full max-w-[1600px] px-6 py-6 lg:px-8 lg:py-8">
    <div class="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
      <aside class="space-y-4 xl:sticky xl:top-24 xl:self-start">
        <section class="surface-card overflow-hidden">
          <div class="border-b border-slate-200 bg-[linear-gradient(135deg,#e0f2fe_0%,#f8fafc_58%,#eef2ff_100%)] px-5 py-5 dark:border-slate-700 dark:bg-[linear-gradient(135deg,rgba(14,165,233,0.22)_0%,rgba(15,23,42,0.86)_58%,rgba(30,41,59,0.96)_100%)]">
            <p class="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700 dark:text-sky-300">System Console</p>
            <h1 class="mt-3 text-2xl font-semibold text-slate-900 dark:text-white">系统设置</h1>
            <p class="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
              用项目现有的 Vue 3、Pinia 和 Tailwind 约定重构后的设置页，统一主题行为和表单外观。
            </p>
          </div>

          <div class="space-y-5 px-5 py-5">
            <ThemeToggleButton :is-dark="themeStore.isDark" :mode="themeStore.mode" @toggle="themeStore.toggleTheme" />

            <div class="rounded-2xl bg-slate-50 p-4 dark:bg-slate-900/70">
              <p class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">当前摘要</p>
              <ul class="mt-3 space-y-2 text-sm text-slate-600 dark:text-slate-300">
                <li v-for="item in quickSummary" :key="item" class="flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-sky-500" />
                  <span>{{ item }}</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        <nav class="surface-card p-3">
          <a
            v-for="item in sectionItems"
            :key="item.id"
            :href="`#${item.id}`"
            class="block rounded-2xl px-4 py-3 transition"
            :class="activeSection === item.id
              ? 'bg-sky-50 text-sky-700 shadow-soft dark:bg-sky-500/10 dark:text-sky-200'
              : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-900 dark:hover:text-white'"
          >
            <p class="text-sm font-medium">{{ item.label }}</p>
            <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">{{ item.description }}</p>
          </a>
        </nav>
      </aside>

      <div class="space-y-6">
        <section class="surface-card overflow-hidden">
          <div class="flex flex-col gap-6 px-6 py-6 lg:flex-row lg:items-end lg:justify-between">
            <div class="max-w-3xl">
              <p class="text-xs font-semibold uppercase tracking-[0.24em] text-sky-600 dark:text-sky-300">Workspace Preferences</p>
              <h2 class="mt-3 text-3xl font-semibold tracking-tight text-slate-900 dark:text-white">统一配置界面、主题模式与工作站偏好</h2>
              <p class="mt-3 text-sm leading-7 text-slate-600 dark:text-slate-300">
                当前页面改为完全基于 Tailwind 实现的表单布局，主题状态交由 Pinia 管理，避免继续依赖未定义的 CSS 变量和独立主题逻辑。
              </p>
            </div>

            <div class="grid gap-3 sm:grid-cols-3">
              <div class="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
                <p class="text-xs uppercase tracking-[0.18em] text-slate-400">Theme</p>
                <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ themeStore.mode }}</p>
              </div>
              <div class="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
                <p class="text-xs uppercase tracking-[0.18em] text-slate-400">Viewer</p>
                <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ defaultLayout }}</p>
              </div>
              <div class="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
                <p class="text-xs uppercase tracking-[0.18em] text-slate-400">DICOM</p>
                <p class="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{{ dicomServerPort }}</p>
              </div>
            </div>
          </div>
        </section>

        <section id="general" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">常规设置</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">设置默认语言、自动保存频率和新建检查的基础偏好。</p>
          </div>

          <div class="grid gap-4 md:grid-cols-2">
            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">界面语言</span>
              <select v-model="language" class="surface-input">
                <option v-for="option in languageOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">自动保存间隔</span>
              <select v-model="autoSaveInterval" class="surface-input">
                <option :value="15">15 秒</option>
                <option :value="30">30 秒</option>
                <option :value="60">60 秒</option>
                <option :value="120">2 分钟</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">默认检查类型</span>
              <select v-model="defaultModality" class="surface-input">
                <option v-for="option in modalityOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">字体大小</span>
              <select v-model="fontSize" class="surface-input">
                <option v-for="option in fontSizeOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>
          </div>
        </section>

        <section id="appearance" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">外观主题</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">主题状态完全接入 Pinia store，与全局 dark class 保持一致。</p>
          </div>

          <div class="grid gap-4 lg:grid-cols-3">
            <button
              v-for="option in themeOptions"
              :key="option.value"
              type="button"
              class="rounded-3xl border p-5 text-left transition"
              :class="themeStore.mode === option.value
                ? 'border-sky-400 bg-sky-50 shadow-soft dark:border-sky-400 dark:bg-sky-500/10'
                : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900/60 dark:hover:border-slate-600 dark:hover:bg-slate-900'"
              @click="setThemeMode(option.value)"
            >
              <p class="text-sm font-semibold text-slate-900 dark:text-white">{{ option.label }}</p>
              <p class="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">{{ option.description }}</p>
              <p class="mt-4 text-xs uppercase tracking-[0.2em] text-slate-400">
                {{ themeStore.mode === option.value ? '当前生效' : '点击切换' }}
              </p>
            </button>
          </div>
        </section>

        <section id="viewer" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">影像查看器</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">为阅片工作流设置默认布局、叠加显示和交互方向。</p>
          </div>

          <div class="grid gap-4 md:grid-cols-2">
            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">默认布局</span>
              <select v-model="defaultLayout" class="surface-input">
                <option v-for="option in layoutOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">影像质量</span>
              <select v-model="imageQuality" class="surface-input">
                <option v-for="option in imageQualityOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">滚动方向</span>
              <select v-model="scrollDirection" class="surface-input">
                <option v-for="option in scrollDirectionOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">测量单位</span>
              <select v-model="measureUnit" class="surface-input">
                <option v-for="option in measureUnitOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
              </select>
            </label>
          </div>

          <div class="mt-6 grid gap-3 sm:grid-cols-3">
            <label class="flex items-center justify-between rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">显示叠加信息</span>
                <span class="mt-1 block text-xs text-slate-500 dark:text-slate-400">患者信息与技术参数</span>
              </span>
              <input v-model="showOverlay" type="checkbox" class="h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>

            <label class="flex items-center justify-between rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">显示十字线</span>
                <span class="mt-1 block text-xs text-slate-500 dark:text-slate-400">多平面重建定位</span>
              </span>
              <input v-model="crosshair" type="checkbox" class="h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>

            <div class="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 dark:border-slate-700 dark:bg-slate-900/70">
              <p class="text-sm font-medium text-slate-800 dark:text-slate-100">当前布局建议</p>
              <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">根据工作站偏好自动预设</p>
              <p class="mt-4 text-xl font-semibold text-slate-900 dark:text-white">{{ defaultLayout }}</p>
            </div>
          </div>
        </section>

        <section id="notifications" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">通知设置</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">定义哪些事件需要在工作台里高优先级提示。</p>
          </div>

          <div class="grid gap-3 md:grid-cols-2">
            <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">危急值通知</span>
                <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">检测到危急值时立即发出通知</span>
              </span>
              <input v-model="notifyCritical" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>

            <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">新病例通知</span>
                <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">有新分配病例时提醒值班医生</span>
              </span>
              <input v-model="notifyNewCase" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>

            <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">报告审核通知</span>
                <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">审核通过或退回时同步更新</span>
              </span>
              <input v-model="notifyReport" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>

            <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
              <span>
                <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">提示音</span>
                <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">通知到达时播放音频提示</span>
              </span>
              <input v-model="soundEnabled" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
            </label>
          </div>
        </section>

        <section id="ai" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">AI 辅助设置</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">配置自动分析策略、阈值和结果高亮方式。</p>
          </div>

          <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div class="space-y-4">
              <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
                <span>
                  <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">自动分析</span>
                  <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">打开影像时自动触发 AI 推理</span>
                </span>
                <input v-model="aiAutoAnalyze" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
              </label>

              <label class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
                <span>
                  <span class="block text-sm font-medium text-slate-800 dark:text-slate-100">高亮发现区域</span>
                  <span class="mt-1 block text-xs leading-6 text-slate-500 dark:text-slate-400">在原图中叠加可疑病灶区域</span>
                </span>
                <input v-model="aiHighlightFindings" type="checkbox" class="mt-1 h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
              </label>
            </div>

            <div class="rounded-3xl border border-slate-200 bg-slate-50 p-5 dark:border-slate-700 dark:bg-slate-900/70">
              <p class="text-sm font-medium text-slate-800 dark:text-slate-100">置信度阈值</p>
              <p class="mt-1 text-xs leading-6 text-slate-500 dark:text-slate-400">低于此值的 AI 建议将被标记为低置信度。</p>
              <input v-model="aiConfidenceThreshold" type="range" min="50" max="99" class="mt-6 w-full accent-blue-600 dark:accent-sky-400" />
              <div class="mt-4 flex items-end justify-between">
                <span class="text-xs uppercase tracking-[0.2em] text-slate-400">当前值</span>
                <span class="text-3xl font-semibold text-slate-900 dark:text-white">{{ aiConfidenceThreshold }}%</span>
              </div>
            </div>
          </div>
        </section>

        <section id="dicom" class="surface-card p-6">
          <div class="mb-6 flex flex-col gap-2 border-b border-slate-200 pb-4 dark:border-slate-700">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">DICOM 服务器连接</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400">维护 PACS 服务的地址、端口与 AE Title 等基础连接参数。</p>
          </div>

          <div class="grid gap-4 md:grid-cols-2">
            <label class="block space-y-2 md:col-span-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">服务器地址</span>
              <input v-model="dicomServerHost" type="text" class="surface-input" placeholder="192.168.1.100" />
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">端口</span>
              <input v-model="dicomServerPort" type="text" class="surface-input" placeholder="4242" />
            </label>

            <label class="block space-y-2">
              <span class="text-sm font-medium text-slate-700 dark:text-slate-200">AE Title</span>
              <input v-model="dicomAeTitle" type="text" class="surface-input" placeholder="MEDIMAGEDX" />
            </label>
          </div>

          <div class="mt-6 flex flex-col gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900/70 md:flex-row md:items-center md:justify-between">
            <div>
              <p class="text-sm font-medium text-slate-800 dark:text-slate-100">自动拉取影像</p>
              <p class="mt-1 text-xs leading-6 text-slate-500 dark:text-slate-400">新检查到达时自动从 PACS 拉取序列和元数据。</p>
            </div>
            <input v-model="dicomAutoFetch" type="checkbox" class="h-5 w-5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 dark:border-slate-600 dark:bg-slate-800 dark:text-sky-400" />
          </div>

          <div class="mt-6 flex justify-end">
            <button type="button" class="surface-button-secondary px-4 py-2.5">测试连接</button>
          </div>
        </section>

        <div class="sticky bottom-4 z-20 rounded-3xl border border-white/70 bg-white/90 p-4 shadow-soft backdrop-blur dark:border-slate-700 dark:bg-slate-950/88">
          <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p class="text-sm font-medium text-slate-900 dark:text-white">设置变更</p>
              <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
                {{ saved ? '设置已保存并同步到当前会话。' : '完成调整后可保存为当前工作站默认配置。' }}
              </p>
            </div>

            <div class="flex flex-wrap items-center gap-3">
              <span
                v-if="saved"
                class="inline-flex items-center rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300"
              >
                已保存
              </span>
              <button type="button" class="surface-button-secondary px-4 py-2.5" @click="resetDefaults">重置默认</button>
              <button type="button" class="surface-button-primary px-5 py-2.5" @click="handleSave">保存设置</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>
