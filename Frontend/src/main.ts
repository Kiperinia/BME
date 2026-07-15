import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

/**
 * brief:
 *   Handle app.
 *
 * parameter:
 *   - None.
 *
 * retrival:
 *   - Returns the computed value or updates local application state.
 */
const app = createApp(App)

app.use(createPinia())
app.use(router)
app.mount('#app')
