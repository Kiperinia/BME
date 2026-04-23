<script setup lang="ts">
import { ref } from 'vue'

/* ---------- Theme ---------- */
const currentTheme = ref(document.documentElement.getAttribute('data-theme') || 'light')
function setTheme(theme: string) {
  currentTheme.value = theme
  document.documentElement.setAttribute('data-theme', theme)
}

/* ---------- General Settings ---------- */
const language = ref('zh-CN')
const autoSaveInterval = ref(30)
const defaultModality = ref('CT')
const imageQuality = ref('high')
const fontSize = ref('medium')

/* ---------- Notification ---------- */
const notifyCritical = ref(true)
const notifyNewCase = ref(true)
const notifyReport = ref(true)
const soundEnabled = ref(false)

/* ---------- Viewer ---------- */
const defaultLayout = ref('2x2')
const showOverlay = ref(true)
const scrollDirection = ref('natural')
const crosshair = ref(false)
const measureUnit = ref('mm')

/* ---------- AI ---------- */
const aiAutoAnalyze = ref(true)
const aiConfidenceThreshold = ref(75)
const aiHighlightFindings = ref(true)

/* ---------- DICOM ---------- */
const dicomServerHost = ref('192.168.1.100')
const dicomServerPort = ref('4242')
const dicomAeTitle = ref('MEDIMAGEDX')
const dicomAutoFetch = ref(true)

/* ---------- Save ---------- */
const saved = ref(false)
function handleSave() {
  saved.value = true
  setTimeout(() => { saved.value = false }, 2000)
}

const layoutOptions = [
  { value: '1x1', label: '1×1 单窗格' },
  { value: '2x2', label: '2×2 四窗格' },
  { value: '1+2', label: '1+2 混合' },
]
</script>

<template>
  <div class="settings-page">
    <div class="settings-header">
      <h1 class="settings-title">系统设置</h1>
      <p class="settings-desc">配置系统参数、显示偏好和连接选项</p>
    </div>

    <div class="settings-body">
      <!-- Left: Section Nav -->
      <nav class="settings-nav">
        <a href="#general" class="settings-nav-item active">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="3" stroke="currentColor" stroke-width="1.5"/><path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.2 4.2l1.4 1.4M14.4 14.4l1.4 1.4M4.2 15.8l1.4-1.4M14.4 5.6l1.4-1.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          常规设置
        </a>
        <a href="#appearance" class="settings-nav-item">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"/><path d="M10 3a7 7 0 000 14V3z" fill="currentColor" opacity="0.15"/></svg>
          外观主题
        </a>
        <a href="#viewer" class="settings-nav-item">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M2 10h16M10 2v16" stroke="currentColor" stroke-width="1.5"/></svg>
          影像查看器
        </a>
        <a href="#notifications" class="settings-nav-item">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><path d="M10 2a5 5 0 00-5 5v3l-1.5 2h13L15 10V7a5 5 0 00-5-5z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M8 16a2 2 0 004 0" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          通知
        </a>
        <a href="#ai" class="settings-nav-item">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"/><path d="M7 10l2 2 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          AI 辅助
        </a>
        <a href="#dicom" class="settings-nav-item">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><rect x="3" y="5" width="14" height="10" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M7 5V3h6v2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="10" cy="10" r="2" stroke="currentColor" stroke-width="1.5"/></svg>
          DICOM 连接
        </a>
      </nav>

      <!-- Right: Settings Forms -->
      <div class="settings-content">
        <!-- General -->
        <section id="general" class="settings-section">
          <h2 class="section-title">常规设置</h2>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">界面语言</label>
              <span class="setting-hint">设置系统显示语言</span>
            </div>
            <select v-model="language" class="setting-select">
              <option value="zh-CN">简体中文</option>
              <option value="en-US">English</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">自动保存间隔</label>
              <span class="setting-hint">报告自动保存的时间间隔（秒）</span>
            </div>
            <select v-model="autoSaveInterval" class="setting-select">
              <option :value="15">15 秒</option>
              <option :value="30">30 秒</option>
              <option :value="60">60 秒</option>
              <option :value="120">2 分钟</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">默认检查类型</label>
              <span class="setting-hint">新建检查时的默认模态</span>
            </div>
            <select v-model="defaultModality" class="setting-select">
              <option value="CT">CT</option>
              <option value="MRI">MRI</option>
              <option value="X-Ray">X-Ray</option>
              <option value="Ultrasound">超声</option>
              <option value="PET">PET</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">字体大小</label>
              <span class="setting-hint">界面字体大小偏好</span>
            </div>
            <select v-model="fontSize" class="setting-select">
              <option value="small">小 (14px)</option>
              <option value="medium">中 (16px)</option>
              <option value="large">大 (18px)</option>
            </select>
          </div>
        </section>

        <!-- Appearance -->
        <section id="appearance" class="settings-section">
          <h2 class="section-title">外观主题</h2>
          <div class="theme-cards">
            <button
              class="theme-card"
              :class="{ active: currentTheme === 'light' }"
              @click="setTheme('light')"
              aria-label="切换到浅色模式"
            >
              <div class="theme-preview theme-light-preview">
                <div class="tp-sidebar"></div>
                <div class="tp-main">
                  <div class="tp-header"></div>
                  <div class="tp-content">
                    <div class="tp-block"></div>
                    <div class="tp-block"></div>
                  </div>
                </div>
              </div>
              <span class="theme-label">浅色模式</span>
            </button>
            <button
              class="theme-card"
              :class="{ active: currentTheme === 'dark' }"
              @click="setTheme('dark')"
              aria-label="切换到深色模式"
            >
              <div class="theme-preview theme-dark-preview">
                <div class="tp-sidebar"></div>
                <div class="tp-main">
                  <div class="tp-header"></div>
                  <div class="tp-content">
                    <div class="tp-block"></div>
                    <div class="tp-block"></div>
                  </div>
                </div>
              </div>
              <span class="theme-label">深色模式</span>
            </button>
          </div>
        </section>

        <!-- Viewer -->
        <section id="viewer" class="settings-section">
          <h2 class="section-title">影像查看器</h2>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">默认布局</label>
              <span class="setting-hint">打开影像时的默认窗格布局</span>
            </div>
            <select v-model="defaultLayout" class="setting-select">
              <option v-for="opt in layoutOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">影像质量</label>
              <span class="setting-hint">影像渲染质量（高质量需要更多带宽）</span>
            </div>
            <select v-model="imageQuality" class="setting-select">
              <option value="low">低质量（快速加载）</option>
              <option value="medium">中等质量</option>
              <option value="high">高质量</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">显示叠加信息</label>
              <span class="setting-hint">在影像上显示患者信息和技术参数</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="showOverlay">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">滚动方向</label>
              <span class="setting-hint">鼠标滚轮切换层面的方向</span>
            </div>
            <select v-model="scrollDirection" class="setting-select">
              <option value="natural">自然方向</option>
              <option value="reverse">反向</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">显示十字线</label>
              <span class="setting-hint">多平面重建时显示十字交叉线</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="crosshair">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">测量单位</label>
              <span class="setting-hint">长度和面积的测量单位</span>
            </div>
            <select v-model="measureUnit" class="setting-select">
              <option value="mm">毫米 (mm)</option>
              <option value="cm">厘米 (cm)</option>
            </select>
          </div>
        </section>

        <!-- Notifications -->
        <section id="notifications" class="settings-section">
          <h2 class="section-title">通知设置</h2>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">危急值通知</label>
              <span class="setting-hint">检测到危急值时立即通知</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="notifyCritical">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">新病例通知</label>
              <span class="setting-hint">有新分配的病例时通知</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="notifyNewCase">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">报告审核通知</label>
              <span class="setting-hint">报告被审核或退回时通知</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="notifyReport">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">提示音</label>
              <span class="setting-hint">收到通知时播放提示音</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="soundEnabled">
              <span class="toggle-slider"></span>
            </label>
          </div>
        </section>

        <!-- AI Settings -->
        <section id="ai" class="settings-section">
          <h2 class="section-title">AI 辅助设置</h2>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">自动分析</label>
              <span class="setting-hint">打开影像时自动启动 AI 分析</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="aiAutoAnalyze">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">置信度阈值</label>
              <span class="setting-hint">低于此值的 AI 建议将被标记为低置信度</span>
            </div>
            <div class="range-group">
              <input type="range" v-model.number="aiConfidenceThreshold" min="50" max="99" class="setting-range">
              <span class="range-value">{{ aiConfidenceThreshold }}%</span>
            </div>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">高亮发现区域</label>
              <span class="setting-hint">在影像上高亮显示 AI 检测到的异常区域</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="aiHighlightFindings">
              <span class="toggle-slider"></span>
            </label>
          </div>
        </section>

        <!-- DICOM -->
        <section id="dicom" class="settings-section">
          <h2 class="section-title">DICOM 服务器连接</h2>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">服务器地址</label>
              <span class="setting-hint">DICOM 服务器 IP 地址或主机名</span>
            </div>
            <input type="text" v-model="dicomServerHost" class="setting-input" placeholder="192.168.1.100">
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">端口</label>
              <span class="setting-hint">DICOM 服务端口号</span>
            </div>
            <input type="text" v-model="dicomServerPort" class="setting-input setting-input-sm" placeholder="4242">
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">AE Title</label>
              <span class="setting-hint">本系统的 Application Entity 标识</span>
            </div>
            <input type="text" v-model="dicomAeTitle" class="setting-input" placeholder="MEDIMAGEDX">
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <label class="setting-label">自动拉取影像</label>
              <span class="setting-hint">接收到检查时自动从 PACS 拉取影像</span>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" v-model="dicomAutoFetch">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row" style="justify-content: flex-end;">
            <button class="btn-outline" style="margin-right: var(--space-3);">测试连接</button>
          </div>
        </section>

        <!-- Save Bar -->
        <div class="save-bar">
          <transition name="fade">
            <span v-if="saved" class="save-indicator">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 8l3 3 5-5" stroke="var(--color-success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
              已保存
            </span>
          </transition>
          <button class="btn-outline">重置默认</button>
          <button class="btn-primary" @click="handleSave">保存设置</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-page {
  padding: var(--space-8);
  max-width: 960px;
  margin: 0 auto;
}

.settings-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0;
}

.settings-desc {
  font-size: 14px;
  color: var(--text-secondary);
  margin: var(--space-2) 0 0;
}

.settings-body {
  display: flex;
  gap: var(--space-8);
  margin-top: var(--space-8);
}

/* Section Navigation */
.settings-nav {
  width: 200px;
  flex-shrink: 0;
  position: sticky;
  top: calc(var(--header-height) + var(--space-8));
  align-self: flex-start;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.settings-nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border-radius: 8px;
  font-size: 14px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 200ms ease;
}

.settings-nav-item:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.settings-nav-item.active {
  background: var(--bg-active);
  color: var(--color-primary);
  font-weight: 600;
}

/* Content */
.settings-content {
  flex: 1;
  min-width: 0;
}

.settings-section {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: var(--space-6);
  margin-bottom: var(--space-6);
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 var(--space-6);
  padding-bottom: var(--space-4);
  border-bottom: 1px solid var(--border-color);
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-4) 0;
  border-bottom: 1px solid var(--border-color);
}

.setting-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.setting-row:first-of-type {
  padding-top: 0;
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.setting-label {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.setting-hint {
  font-size: 12px;
  color: var(--text-muted);
}

.setting-select,
.setting-input {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-input);
  color: var(--text-primary);
  font-size: 14px;
  min-width: 180px;
  outline: none;
  transition: border-color 200ms ease;
}

.setting-select:focus,
.setting-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

.setting-input-sm {
  min-width: 100px;
  max-width: 120px;
}

/* Toggle */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 44px;
  height: 24px;
  cursor: pointer;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  inset: 0;
  background: var(--border-color);
  border-radius: 24px;
  transition: background 200ms ease;
}

.toggle-slider::before {
  content: '';
  position: absolute;
  left: 2px;
  top: 2px;
  width: 20px;
  height: 20px;
  background: #fff;
  border-radius: 50%;
  transition: transform 200ms ease;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
}

.toggle-switch input:checked + .toggle-slider {
  background: var(--color-primary);
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

.toggle-switch input:focus-visible + .toggle-slider {
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
}

/* Range */
.range-group {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.setting-range {
  width: 140px;
  accent-color: var(--color-primary);
}

.range-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-primary);
  min-width: 40px;
  text-align: right;
}

/* Theme Cards */
.theme-cards {
  display: flex;
  gap: var(--space-4);
}

.theme-card {
  flex: 1;
  border: 2px solid var(--border-color);
  border-radius: 12px;
  padding: var(--space-4);
  background: var(--bg-input);
  cursor: pointer;
  transition: all 200ms ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-3);
}

.theme-card:hover {
  border-color: var(--color-primary);
}

.theme-card.active {
  border-color: var(--color-primary);
  background: var(--bg-active);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

.theme-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

/* Theme preview mini mockups */
.theme-preview {
  width: 100%;
  height: 80px;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
}

.theme-light-preview { background: #f1f5f9; }
.theme-light-preview .tp-sidebar { background: #fff; width: 24%; border-right: 1px solid #e2e8f0; }
.theme-light-preview .tp-main { flex: 1; display: flex; flex-direction: column; }
.theme-light-preview .tp-header { height: 16px; background: #fff; border-bottom: 1px solid #e2e8f0; }
.theme-light-preview .tp-content { flex: 1; display: flex; gap: 4px; padding: 4px; }
.theme-light-preview .tp-block { flex: 1; background: #fff; border-radius: 4px; }

.theme-dark-preview { background: #0f172a; }
.theme-dark-preview .tp-sidebar { background: #1e293b; width: 24%; border-right: 1px solid #334155; }
.theme-dark-preview .tp-main { flex: 1; display: flex; flex-direction: column; }
.theme-dark-preview .tp-header { height: 16px; background: #1e293b; border-bottom: 1px solid #334155; }
.theme-dark-preview .tp-content { flex: 1; display: flex; gap: 4px; padding: 4px; }
.theme-dark-preview .tp-block { flex: 1; background: #1e293b; border-radius: 4px; }

/* Buttons */
.btn-primary {
  padding: var(--space-2) var(--space-6);
  background: var(--color-primary);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 200ms ease;
}

.btn-primary:hover {
  background: var(--color-primary-hover);
}

.btn-outline {
  padding: var(--space-2) var(--space-4);
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: all 200ms ease;
}

.btn-outline:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* Save Bar */
.save-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-3);
  padding: var(--space-6) 0;
  position: sticky;
  bottom: 0;
  background: var(--bg-body);
}

.save-indicator {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: 13px;
  color: var(--color-success);
  font-weight: 500;
  margin-right: auto;
}

/* Responsive */
@media (max-width: 768px) {
  .settings-body {
    flex-direction: column;
  }
  .settings-nav {
    position: static;
    width: 100%;
    flex-direction: row;
    flex-wrap: wrap;
  }
}
</style>
