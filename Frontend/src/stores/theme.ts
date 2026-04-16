import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export type ThemeMode = 'light' | 'dark' | 'system'

const themeStorageKey = 'bme-theme-mode'
const prefersDarkMediaQuery = '(prefers-color-scheme: dark)'

let systemPreferenceListenerBound = false

export const useThemeStore = defineStore('theme', () => {
  const mode = ref<ThemeMode>('system')
  const systemDark = ref(false)

  const resolvedTheme = computed<'light' | 'dark'>(() => {
    if (mode.value === 'system') {
      return systemDark.value ? 'dark' : 'light'
    }

    return mode.value
  })

  const isDark = computed(() => resolvedTheme.value === 'dark')

  const applyTheme = () => {
    if (typeof document === 'undefined') {
      return
    }

    document.documentElement.classList.toggle('dark', isDark.value)
    document.documentElement.style.colorScheme = resolvedTheme.value
  }

  const setMode = (nextMode: ThemeMode) => {
    mode.value = nextMode
    window.localStorage.setItem(themeStorageKey, nextMode)
    applyTheme()
  }

  const initializeTheme = () => {
    if (typeof window === 'undefined') {
      return
    }

    const mediaQuery = window.matchMedia(prefersDarkMediaQuery)
    systemDark.value = mediaQuery.matches

    if (!systemPreferenceListenerBound) {
      mediaQuery.addEventListener('change', (event) => {
        systemDark.value = event.matches
        applyTheme()
      })
      systemPreferenceListenerBound = true
    }

    const storedTheme = window.localStorage.getItem(themeStorageKey)
    if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') {
      mode.value = storedTheme
    }

    applyTheme()
  }

  const toggleTheme = () => {
    setMode(isDark.value ? 'light' : 'dark')
  }

  const resetToSystem = () => {
    setMode('system')
  }

  return {
    mode,
    resolvedTheme,
    isDark,
    initializeTheme,
    setMode,
    toggleTheme,
    resetToSystem,
  }
})