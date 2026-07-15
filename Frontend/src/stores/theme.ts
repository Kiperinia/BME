import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export type ThemeMode = 'light' | 'dark' | 'system'

/**
 * brief:
 *   Handle theme storage key.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const themeStorageKey = 'bme-theme-mode'
/**
 * brief:
 *   Handle prefers dark media query.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const prefersDarkMediaQuery = '(prefers-color-scheme: dark)'

let systemPreferenceListenerBound = false

/**
 * brief:
 *   Handle use theme store.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
export const useThemeStore = defineStore('theme', () => {
  /**
   * brief:
   *   Handle mode.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const mode = ref<ThemeMode>('system')
  /**
   * brief:
   *   Handle system dark.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const systemDark = ref(false)

  /**
   * brief:
   *   Handle resolved theme.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const resolvedTheme = computed<'light' | 'dark'>(() => {
    if (mode.value === 'system') {
      return systemDark.value ? 'dark' : 'light'
    }

    return mode.value
  })

  /**
   * brief:
   *   Handle is dark.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const isDark = computed(() => resolvedTheme.value === 'dark')

  /**
   * brief:
   *   Apply theme.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const applyTheme = () => {
    if (typeof document === 'undefined') {
      return
    }

    document.documentElement.classList.toggle('dark', isDark.value)
    document.documentElement.style.colorScheme = resolvedTheme.value
  }

  /**
   * brief:
   *   Handle set mode.
   *
   * parameter:
   *   - nextMode: Input value for nextMode.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const setMode = (nextMode: ThemeMode) => {
    mode.value = nextMode
    window.localStorage.setItem(themeStorageKey, nextMode)
    applyTheme()
  }

  /**
   * brief:
   *   Handle initialize theme.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const initializeTheme = () => {
    if (typeof window === 'undefined') {
      return
    }

    /**
     * brief:
     *   Handle media query.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const mediaQuery = window.matchMedia(prefersDarkMediaQuery)
    systemDark.value = mediaQuery.matches

    if (!systemPreferenceListenerBound) {
      mediaQuery.addEventListener('change', (event) => {
        systemDark.value = event.matches
        applyTheme()
      })
      systemPreferenceListenerBound = true
    }

    /**
     * brief:
     *   Handle stored theme.
     *
     * parameter:
     *   - None.
     *
     * retrival:
     *   - Returns the computed value or updates local application state.
     */
    const storedTheme = window.localStorage.getItem(themeStorageKey)
    if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') {
      mode.value = storedTheme
    }

    applyTheme()
  }

  /**
   * brief:
   *   Toggle theme.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
  const toggleTheme = () => {
    setMode(isDark.value ? 'light' : 'dark')
  }

  /**
   * brief:
   *   Reset to system.
   *
   * parameter:
   *   - None.
   *
   * retrival:
   *   - Returns the computed value or updates local application state.
   */
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
