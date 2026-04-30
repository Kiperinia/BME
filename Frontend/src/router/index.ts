import { createRouter, createWebHashHistory } from 'vue-router'

import ReportBuilderPage from '@/pages/ReportBuilderPage.vue'
import ReportGenerationPage from '@/pages/ReportGenerationPage.vue'
import SystemSettings from '@/pages/SystemSettings.vue'

const router = createRouter({
	history: createWebHashHistory(),
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
				title: '检查工作台',
			},
		},
		{
			path: '/report-generation',
			name: 'report-generation',
			component: ReportGenerationPage,
			meta: {
				title: '智能报告生成',
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
