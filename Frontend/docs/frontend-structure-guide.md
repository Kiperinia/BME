# Frontend 目录结构与文件组织说明

本文聚焦 Frontend 目录，说明当前各文件夹的职责、常见文件的组织方式，以及后续继续扩展时建议遵循的放置规则。

## 1. Frontend 根目录是做什么的

Frontend 是一个基于 Vue 3 + Vite + TypeScript + Pinia + Tailwind CSS 的前端工程。

当前根目录大致可以分成 5 类内容：

1. 应用源码：src
2. 静态资源：public
3. 文档：docs
4. 工程配置：package.json、tsconfig、vite.config.ts、tailwind.config.ts 等
5. 构建产物和依赖：dist、node_modules

一个接近当前实际情况的结构可以概括为：

```text
Frontend/
  .env.example
  .gitignore
  env.d.ts
  index.html
  package.json
  package-lock.json
  postcss.config.cjs
  tailwind.config.ts
  tsconfig.json
  tsconfig.app.json
  tsconfig.node.json
  vite.config.ts
  README.md
  .vscode/
  dist/
  docs/
  public/
  src/
```

## 2. 根目录下各文件夹的含义

### src

这是前端最核心的业务代码目录。页面、组件、状态管理、类型定义、API 封装基本都放在这里。

当前这个项目的主要开发重心就在 src 内。

### public

这里放不会经过模块打包处理、但需要被浏览器直接访问的静态资源。

当前项目里主要是 images 目录，适合存放：

1. 演示图片
2. 默认占位图
3. 报告构建流程里的本地图像资源

这类资源在代码里通常以绝对路径引用，例如 /images/xxx.svg。

### docs

这里放项目内部文档，而不是业务代码。

当前已经有：

1. report-builder-spec.md：报告构建器的接口、组件、数据结构设计说明
2. frontend-structure-guide.md：本说明文档

适合继续放在 docs 的内容包括：

1. 页面设计说明
2. API 对接约定
3. 状态流说明
4. 组件拆分规则

### .vscode

这是 VS Code 工作区配置目录，通常存放编辑器推荐扩展或局部开发配置，不属于运行时代码。

### dist

这是前端构建产物目录，通常由 npm run build 生成，不应该手工修改。

### node_modules

这是依赖安装目录，由 npm install 生成，不应该手工修改。

## 3. src 目录内部怎么组织

当前 src 目录结构如下：

```text
src/
  api/
  assets/
  components/
    common/
    icons/
    layout/
    report/
  composables/
  pages/
  router/
  stores/
  types/
  views/
  App.vue
  main.ts
```

### main.ts

这是应用入口文件。

当前职责很明确：

1. 引入全局样式 assets/main.css
2. 创建 Vue 应用
3. 注册 Pinia
4. 挂载到 index.html 里的 #app 节点

这种文件通常保持很薄，只做启动和注册，不直接写业务逻辑。

### App.vue

这是根组件，相当于整个前端应用的总装配层。

当前项目里，App.vue 直接承担了报告工作台的主要页面编排和状态协调，导入了：

1. 布局组件
2. 通用组件
3. report 业务组件
4. Pinia store
5. API mock 方法
6. 类型定义

从组织角度看，App.vue 目前更像“页面级容器”。如果未来页面继续增加，可以把更多页面层逻辑下沉到 views，再让 App.vue 主要负责路由出口。

### api

这个目录用于封装接口访问层。

当前已有 reportBuilder.ts，它负责：

1. 创建 axios 实例
2. 维护接口契约定义
3. 提供页面调用的异步函数
4. 在当前阶段承载 mock 数据与 mock 请求逻辑
5. 做前后端字段映射，例如下划线字段转 camelCase

这里适合放的文件形式通常是：

1. 以业务域命名的 API 文件，例如 reportBuilder.ts、patient.ts、auth.ts
2. 每个文件导出若干请求函数
3. 如有需要，单独抽一个 http.ts 放通用 axios 实例

当前项目已经采用了“按业务域拆分 API 文件”的组织方式。

### assets

这个目录用于存放会被构建工具处理的前端资源。

当前已有：

1. base.css：基础样式
2. main.css：全局样式入口
3. logo.svg：示例图标资源

这里和 public 的区别是：

1. assets 中的资源属于模块系统的一部分
2. public 中的资源更适合直接按 URL 引用

如果是全局 CSS、主题变量、组件级共享图标，优先考虑放 assets。

### components

这个目录用于放可复用组件。当前拆分方式比较清晰，已经按职责分层。

#### components/common

放通用业务组件或跨页面可复用组件，例如：

1. EndoVideoPlayer.vue
2. FeedbackToast.vue
3. PatientInfoCard.vue
4. SmartAnnotationTags.vue
5. ThemeToggleButton.vue
6. TumorMaskViewer.vue

这一层的特点是：

1. 功能相对独立
2. 被多个页面或多个业务区块复用的可能性高
3. 更偏“功能组件”而不是整页拼装

#### components/layout

放页面骨架或布局壳组件，例如：

1. WorkspaceHeaderShell.vue
2. WorkspaceSidebarShell.vue

这类组件主要负责：

1. 版式结构
2. 插槽容器
3. 区域分栏

通常不直接承担复杂业务规则。

#### components/report

放与“报告构建”这个业务域强相关的组件，例如：

1. CaptureContextPanel.vue
2. ReportPreviewPanel.vue
3. TumorInfoPanel.vue
4. WorkspaceControlPanel.vue

这一层属于“垂直业务组件”，和 common 的区别是复用范围更窄，但业务语义更强。

#### components/icons

放图标组件，通常每个文件对应一个图标组件，例如 IconCommunity.vue。

这种组织方式的优点是：

1. 统一图标使用方式
2. 可以直接像组件一样传 class 或样式
3. 便于按需替换 SVG 实现

#### components 根目录下的 HelloWorld.vue、TheWelcome.vue、WelcomeItem.vue

这些更像 Vue 初始化模板遗留文件，主要用于示例，不是当前业务核心。

如果项目后续继续业务化，可以考虑：

1. 删除不用的模板示例组件
2. 保留纯业务相关组件结构

### composables

这个目录当前是空的。

它的用途通常是放组合式函数，也就是可复用的逻辑抽取层，例如：

1. useTheme.ts
2. useReportDraft.ts
3. useVideoCapture.ts

适合放进 composables 的逻辑特点：

1. 不直接渲染 UI
2. 关注状态组合与行为复用
3. 多个组件可能共享

如果以后 App.vue 中的状态逻辑继续增长，优先考虑把可复用逻辑下沉到这里。

### pages

这个目录当前有 SystemSettings.vue。

pages 一般表示“独立页面实体”，通常对应一个完整页面，而不是页面中的某个区块。

适合放在 pages 的文件：

1. 设置页
2. 登录页
3. 工作台页
4. 列表页

如果后续启用 Vue Router，pages 往往会成为路由组件的主要来源之一。

### views

这个目录当前有 ReportBuilderView.vue。

views 和 pages 在很多项目里语义接近，但当前项目里更适合这样理解：

1. views 偏“业务视图容器”
2. pages 偏“完整页面入口”

从现有实现看，ReportBuilderView.vue 负责：

1. 组织报告构建页面的数据状态
2. 调用 mock API
3. 协调多个展示组件
4. 对外通过 props 和 emits 暴露能力

因此 views 更像“带业务编排能力的容器组件”。

### router

这个目录当前已有 index.ts，但文件是空的。

它的预期用途很明确：

1. 定义路由表
2. 注册 Vue Router
3. 管理页面切换入口

也就是说，目录已经预留，但当前项目还没有正式启用路由。

### stores

这个目录用于 Pinia 状态管理。

当前已有 theme.ts，用于：

1. 管理 light、dark、system 三种主题模式
2. 监听系统主题变化
3. 持久化到 localStorage
4. 直接控制 document 根节点样式类

这里建议延续“一个领域一个 store 文件”的方式，例如：

1. theme.ts
2. reportDraft.ts
3. patient.ts

### types

这个目录用于统一维护 TypeScript 类型定义。

当前已有 eis.ts，包含：

1. 患者数据类型
2. 视频帧与 mask 类型
3. 报告上下文类型
4. 请求响应类型
5. API 契约类型

这个目录的意义很大，因为它把：

1. 组件 props 的结构
2. API 请求响应格式
3. 前后端接口边界

统一收敛到了一个地方。

后续建议按业务域继续拆分，例如：

1. report.ts
2. patient.ts
3. common.ts

## 4. 根目录下各类配置文件是什么意思

### package.json

这是前端工程的依赖与脚本入口。

当前可以看出项目使用了：

1. vue
2. pinia
3. axios
4. vite
5. typescript
6. tailwindcss

脚本组织也很典型：

1. npm run dev：开发启动
2. npm run build：类型检查加生产构建
3. npm run preview：预览构建结果

### vite.config.ts

这是 Vite 配置文件。

当前主要做了两件事：

1. 注册 Vue 插件和 Vue DevTools 插件
2. 配置 @ 指向 src 的别名

所以项目里大量 import 会写成 @/components/...，而不是相对路径层层回退。

### tailwind.config.ts

这是 Tailwind CSS 配置文件。

当前项目里定义了：

1. 扫描范围 content
2. darkMode 使用 class 模式
3. 自定义阴影和动画

这说明样式体系已经是“Tailwind 原子类 + 少量主题扩展”的方式。

### postcss.config.cjs

这是 PostCSS 配置文件，当前启用了：

1. tailwindcss
2. autoprefixer

文件后缀是 cjs，说明它采用 CommonJS 导出配置对象。

### tsconfig.json

这是 TypeScript 顶层配置聚合文件。当前它主要引用：

1. tsconfig.app.json
2. tsconfig.node.json

也就是说，浏览器端代码和 Node 环境配置是分开的。

### tsconfig.app.json

这是给 src 应用代码使用的 TS 配置。

当前重点包括：

1. 包含 env.d.ts 与 src 下的 ts/vue 文件
2. 配置 @/* 到 src/* 的路径别名
3. 开启 noUncheckedIndexedAccess 这类更严格的类型检查

### tsconfig.node.json

这是给 Vite 配置文件等 Node 环境文件使用的 TS 配置。

它和 tsconfig.app.json 分开，是因为运行环境不同。

### env.d.ts

这个文件用于给 import.meta.env 提供类型声明。

当前已经声明了：

1. VITE_API_BASE_URL
2. VITE_AGENT_API_BASE_URL

这能避免在代码里读取环境变量时丢失类型提示。

### .env.example

这是环境变量示例文件，告诉开发者运行前需要提供哪些变量。

当前前端主要依赖两个接口前缀：

1. VITE_API_BASE_URL
2. VITE_AGENT_API_BASE_URL

### index.html

这是 Vite 应用的 HTML 外壳文件，前端最终会把 Vue 应用挂载到这里的 app 节点。

### .gitignore

用于声明哪些文件不进入 Git 版本管理。当前主要忽略：

1. 日志
2. node_modules
3. dist
4. 编辑器缓存
5. tsbuildinfo
6. 测试产物

## 5. 当前项目里常见文件类型的组织格式

### .vue 文件

当前项目的 .vue 文件主要有三类：

1. 根组件，例如 App.vue
2. 页面或视图组件，例如 ReportBuilderView.vue、SystemSettings.vue
3. 可复用组件，例如 common、layout、report、icons 下的组件

组织上普遍采用单文件组件格式，也就是：

```vue
<script setup lang="ts">
// 状态、计算属性、事件处理、API 调用
</script>

<template>
  <!-- 结构 -->
</template>

<style scoped>
/* 如有局部样式 */
</style>
```

当前项目里更偏向：

1. 使用 script setup
2. 使用 TypeScript
3. 样式大量依赖 Tailwind 类名
4. 尽量通过 props 和 emits 建立组件边界

### .ts 文件

当前 src 内的 .ts 文件主要有四种角色：

1. 启动文件，例如 main.ts
2. API 文件，例如 api/reportBuilder.ts
3. store 文件，例如 stores/theme.ts
4. 类型文件，例如 types/eis.ts

这些文件一般遵循：

1. 一个文件只负责一个明确领域
2. 以命名导出为主
3. 类型和实现适度分离

### .css 文件

当前主要放在 assets 下，通常分为：

1. 基础样式
2. 全局样式入口
3. 主题或变量扩展

项目已经在使用 Tailwind，因此 CSS 文件更适合承载：

1. 全局变量
2. reset 或 base 层
3. 少量公共 class

### .md 文件

Markdown 文件主要放在 docs 或 README.md 中，用于：

1. 设计说明
2. 开发规范
3. 接口契约
4. 项目结构说明

### .d.ts 文件

这类文件用于类型声明补充。当前 env.d.ts 的作用，就是把 Vite 环境变量变成有类型约束的字段。

### .json 文件

当前包括 package-lock.json 以及 tsconfig 系列配置。主要作用是：

1. 依赖锁定
2. TypeScript 编译配置
3. 工程化参数声明

### .cjs 文件

当前是 postcss.config.cjs，用 CommonJS 语法导出配置，常见于部分工具链仍采用 CJS 读取配置的场景。

## 6. 当前 Frontend 的组织特点总结

这个前端项目目前有几个很明显的组织特征：

1. 技术栈标准，采用 Vue 3 + Vite + TypeScript + Pinia + Tailwind
2. src 下已经按职责做了分层，不是把所有文件堆在一起
3. components 进一步按 common、layout、report、icons 细分，说明组件层次意识比较清楚
4. API、类型、状态管理已经各自独立成目录，后续扩展成本较低
5. router 和 composables 已预留，但目前仍有继续完善空间
6. App.vue 当前承担了较多页面编排职责，后续可以逐步向 views 和 composables 拆分

## 7. 后续扩展时建议遵循的规则

为了让 Frontend 目录继续保持清晰，建议后续新增文件时遵循下面的原则：

1. 页面级文件优先放 pages 或 views，不要直接堆到 components
2. 强业务组件放 components/report 一类的垂直目录
3. 通用组件放 components/common
4. 可复用逻辑放 composables，而不是一直堆在 App.vue
5. API 调用与 mock 数据优先放 api，不要散落在页面组件中
6. 类型统一沉淀到 types，避免同一结构在多个文件重复声明
7. 配置文件留在根目录，便于工程入口统一查找

如果你后面想继续整理，我建议下一步可以做两件事：

1. 明确 pages 和 views 的边界，只保留一种主页面组织方式
2. 把 App.vue 里的业务状态抽到 composables 或 views，减少根组件负担