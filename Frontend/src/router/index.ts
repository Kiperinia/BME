import { createRouter, createWebHistory } from 'vue-router'

import ReportBuilderPage from '@/pages/ReportBuilderPage.vue'
import SystemSettings from '@/pages/SystemSettings.vue'

const router = createRouter({
	history: createWebHistory(),
	routes: [
		{
			path: '/',
			redirect: '/report-builder',
		},
		{
			path: '/report-builder',
			name: 'report-builder',
			component: ReportBuilderPage,
			meta: {
				title: '报告构建工作台',
			},
		},
		{
			path: '/settings',
			name: 'settings',
			component: SystemSettings,
			meta: {
				title: '系统设置',
			},
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
