<script setup lang="ts">
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import MarkdownIt from 'markdown-it'

const props = defineProps<{
  markdown: string
}>()

const markdown = new MarkdownIt({
  breaks: true,
  html: false,
  linkify: true,
  typographer: true,
})

const renderedHtml = computed(() => {
  const rawHtml = markdown.render(props.markdown || '')
  return DOMPurify.sanitize(rawHtml)
})
</script>

<template>
  <article class="markdown-card rounded-[28px] border border-slate-200 bg-white px-6 py-6 shadow-soft dark:border-slate-700 dark:bg-slate-950">
    <div class="markdown-body" v-html="renderedHtml" />
  </article>
</template>

<style scoped>
.markdown-body {
  color: rgb(51 65 85);
  line-height: 1.8;
  font-size: 0.98rem;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  color: rgb(15 23 42);
  font-weight: 700;
  line-height: 1.35;
  margin-top: 1.4rem;
  margin-bottom: 0.8rem;
}

.markdown-body :deep(h1) {
  font-size: 1.75rem;
  border-bottom: 1px solid rgb(226 232 240);
  padding-bottom: 0.75rem;
  margin-top: 0;
}

.markdown-body :deep(h2) {
  font-size: 1.25rem;
}

.markdown-body :deep(h3) {
  font-size: 1.05rem;
}

.markdown-body :deep(p),
.markdown-body :deep(blockquote),
.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0.8rem 0;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: 1.4rem;
}

.markdown-body :deep(li) {
  margin: 0.3rem 0;
}

.markdown-body :deep(blockquote) {
  border-left: 4px solid rgb(125 211 252);
  background: rgb(240 249 255);
  border-radius: 0.75rem;
  padding: 0.8rem 1rem;
  color: rgb(14 116 144);
}

.markdown-body :deep(code) {
  background: rgb(241 245 249);
  border-radius: 0.4rem;
  padding: 0.08rem 0.35rem;
  font-size: 0.9em;
}

.markdown-body :deep(pre) {
  background: rgb(15 23 42);
  color: rgb(226 232 240);
  border-radius: 1rem;
  padding: 1rem;
  overflow-x: auto;
}

.markdown-body :deep(pre code) {
  background: transparent;
  padding: 0;
  color: inherit;
}

.markdown-body :deep(strong) {
  color: rgb(15 23 42);
  font-weight: 700;
}

.markdown-body :deep(hr) {
  border: 0;
  border-top: 1px solid rgb(226 232 240);
  margin: 1.2rem 0;
}

:global(.dark) .markdown-body {
  color: rgb(203 213 225);
}

:global(.dark) .markdown-body :deep(h1),
:global(.dark) .markdown-body :deep(h2),
:global(.dark) .markdown-body :deep(h3),
:global(.dark) .markdown-body :deep(strong) {
  color: rgb(248 250 252);
}

:global(.dark) .markdown-body :deep(blockquote) {
  background: rgba(12, 74, 110, 0.35);
  color: rgb(186 230 253);
}

:global(.dark) .markdown-body :deep(code) {
  background: rgb(30 41 59);
}

:global(.dark) .markdown-body :deep(h1),
:global(.dark) .markdown-body :deep(hr) {
  border-color: rgb(51 65 85);
}
</style>
