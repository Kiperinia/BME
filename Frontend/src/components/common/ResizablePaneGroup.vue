<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

type PaneOrientation = 'horizontal' | 'vertical'

const props = withDefaults(defineProps<{
	storageKey: string
	paneIds: string[]
	orientation?: PaneOrientation
	defaultSizes?: number[]
	minSizes?: number[]
	handleThickness?: number
	collapseBelow?: number
	collapsedOrientation?: PaneOrientation
}>(), {
	orientation: 'horizontal',
	defaultSizes: () => [],
	minSizes: () => [],
	handleThickness: 12,
	collapseBelow: 0,
	collapsedOrientation: 'vertical',
})

const containerRef = ref<HTMLElement>()
const paneSizes = ref<number[]>([])
const viewportWidth = ref(typeof window === 'undefined' ? 1440 : window.innerWidth)

const totalHandleThickness = computed(() => {
	return Math.max(0, props.paneIds.length - 1) * props.handleThickness
})

const activeOrientation = computed<PaneOrientation>(() => {
	if (props.collapseBelow > 0 && viewportWidth.value < props.collapseBelow) {
		return props.collapsedOrientation
	}

	return props.orientation
})

const minimumSizes = computed(() => {
	return props.paneIds.map((_, index) => {
		return Math.max(8, props.minSizes[index] ?? 12)
	})
})

const buildEvenSizes = (count: number) => {
	if (count <= 0) {
		return []
	}

	return Array.from({ length: count }, () => 100 / count)
}

const normalizeSizes = (sizes: number[], count: number) => {
	if (count <= 0) {
		return []
	}

	const source = sizes.length === count ? sizes : buildEvenSizes(count)
	const sanitized = source.map((value) => (Number.isFinite(value) && value > 0 ? value : 0))
	const total = sanitized.reduce((sum, value) => sum + value, 0)

	if (total <= 0) {
		return buildEvenSizes(count)
	}

	return sanitized.map((value) => (value / total) * 100)
}

const persistSizes = () => {
	if (typeof window === 'undefined' || !props.storageKey || !paneSizes.value.length) {
		return
	}

	window.localStorage.setItem(props.storageKey, JSON.stringify(paneSizes.value))
}

const hydrateSizes = () => {
	const paneCount = props.paneIds.length
	const fallback = normalizeSizes(props.defaultSizes, paneCount)

	if (typeof window === 'undefined' || !props.storageKey) {
		paneSizes.value = fallback
		return
	}

	const cachedValue = window.localStorage.getItem(props.storageKey)
	if (!cachedValue) {
		paneSizes.value = fallback
		return
	}

	try {
		const parsed = JSON.parse(cachedValue)
		paneSizes.value = normalizeSizes(Array.isArray(parsed) ? parsed : fallback, paneCount)
	} catch {
		paneSizes.value = fallback
	}
}

const paneStyle = (index: number) => {
	const size = paneSizes.value[index] ?? 0
	const minimum = minimumSizes.value[index] ?? 8
	const handleShare = (size / 100) * totalHandleThickness.value
	const minimumHandleShare = (minimum / 100) * totalHandleThickness.value

	if (activeOrientation.value === 'horizontal') {
		return {
			flex: `0 0 calc(${size}% - ${handleShare}px)`,
			minWidth: `calc(${minimum}% - ${minimumHandleShare}px)`,
			minHeight: '0',
			height: '100%',
		}
	}

	return {
		flex: `0 0 calc(${size}% - ${handleShare}px)`,
		minHeight: `calc(${minimum}% - ${minimumHandleShare}px)`,
		minWidth: '0',
		width: '100%',
	}
}

const updateViewportWidth = () => {
	if (typeof window === 'undefined') {
		return
	}

	viewportWidth.value = window.innerWidth
}

const startResize = (paneIndex: number, event: PointerEvent) => {
	if (!containerRef.value || paneIndex >= paneSizes.value.length - 1) {
		return
	}

	event.preventDefault()

	const containerRect = containerRef.value.getBoundingClientRect()
	const totalSize = activeOrientation.value === 'horizontal' ? containerRect.width : containerRect.height
	if (totalSize <= 0) {
		return
	}

	const startPosition = activeOrientation.value === 'horizontal' ? event.clientX : event.clientY
	const startPrevious = paneSizes.value[paneIndex] ?? 0
	const startNext = paneSizes.value[paneIndex + 1] ?? 0
	const pairTotal = startPrevious + startNext
	const minPrevious = minimumSizes.value[paneIndex] ?? 8
	const minNext = minimumSizes.value[paneIndex + 1] ?? 8

	const originalCursor = document.body.style.cursor
	const originalUserSelect = document.body.style.userSelect
	document.body.style.cursor = activeOrientation.value === 'horizontal' ? 'col-resize' : 'row-resize'
	document.body.style.userSelect = 'none'

	const handlePointerMove = (moveEvent: PointerEvent) => {
		const currentPosition = activeOrientation.value === 'horizontal' ? moveEvent.clientX : moveEvent.clientY
		const deltaPercent = ((currentPosition - startPosition) / totalSize) * 100
		const nextPrevious = Math.min(Math.max(startPrevious + deltaPercent, minPrevious), pairTotal - minNext)
		const nextPaneSizes = [...paneSizes.value]

		nextPaneSizes[paneIndex] = nextPrevious
		nextPaneSizes[paneIndex + 1] = pairTotal - nextPrevious
		paneSizes.value = normalizeSizes(nextPaneSizes, props.paneIds.length)
	}

	const handlePointerUp = () => {
		persistSizes()
		document.body.style.cursor = originalCursor
		document.body.style.userSelect = originalUserSelect
		window.removeEventListener('pointermove', handlePointerMove)
		window.removeEventListener('pointerup', handlePointerUp)
	}

	window.addEventListener('pointermove', handlePointerMove)
	window.addEventListener('pointerup', handlePointerUp)
}

watch(
	() => [props.storageKey, props.paneIds.join('|'), props.defaultSizes.join('|')],
	() => {
		hydrateSizes()
	},
	{ immediate: true },
)

watch(
	() => paneSizes.value,
	() => {
		if (paneSizes.value.length) {
			persistSizes()
		}
	},
	{ deep: true },
)

onMounted(() => {
	updateViewportWidth()
	window.addEventListener('resize', updateViewportWidth)
})

onBeforeUnmount(() => {
	if (typeof window !== 'undefined') {
		window.removeEventListener('resize', updateViewportWidth)
	}
})
</script>

<template>
	<div
		ref="containerRef"
		class="flex min-h-0 min-w-0"
		:class="activeOrientation === 'horizontal' ? 'flex-row items-stretch' : 'flex-col items-stretch'"
	>
		<template v-for="(paneId, paneIndex) in paneIds" :key="paneId">
			<div class="min-h-0 min-w-0 overflow-hidden" :style="paneStyle(paneIndex)">
				<slot :name="paneId" />
			</div>

			<button
				v-if="paneIndex < paneIds.length - 1"
				type="button"
				class="group relative flex flex-none items-center justify-center rounded-full bg-transparent transition hover:bg-sky-100/70 dark:hover:bg-sky-950/40"
				:class="activeOrientation === 'horizontal'
					? 'mx-1 h-full w-3 cursor-col-resize'
					: 'my-1 h-3 w-full cursor-row-resize'"
				:aria-label="activeOrientation === 'horizontal' ? '调整左右分栏宽度' : '调整上下分栏高度'"
				@pointerdown="startResize(paneIndex, $event)"
			>
				<span
					class="rounded-full bg-slate-300 transition group-hover:bg-sky-500 dark:bg-slate-700 dark:group-hover:bg-sky-400"
					:class="activeOrientation === 'horizontal' ? 'h-12 w-1' : 'h-1 w-12'"
				/>
			</button>
		</template>
	</div>
</template>
