import { createRouter, createWebHashHistory } from 'vue-router'

import PatientRecordsPage from '@/pages/PatientRecordsPage.vue'
import SystemSettings from '@/pages/SystemSettings.vue'
import WorkspacePage from '@/pages/WorkspacePage.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/',
      redirect: '/workspace',
    },
    {
      path: '/workspace',
      name: 'workspace',
      component: WorkspacePage,
      meta: {
        title: 'Workspace',
      },
    },
    {
      path: '/patients',
      name: 'patients',
      component: PatientRecordsPage,
      meta: {
        title: 'Patient Records',
      },
    },
    {
      path: '/settings',
      name: 'settings',
      component: SystemSettings,
      meta: {
        title: 'Settings',
      },
    },
    {
      path: '/report-builder',
      redirect: '/workspace',
    },
    {
      path: '/report-generation',
      redirect: '/workspace',
    },
  ],
})

router.afterEach((to) => {
  if (typeof document === 'undefined') {
    return
  }

  document.title = `${to.meta.title ?? 'BME Frontend'} - BME`
})

export default router
