# EIS Report Builder Spec

## API 接口定义

### 1. 智能生成报告草稿
- URL: `${VITE_AGENT_API_BASE_URL}/v1/report-drafts/generate`
- Method: `POST`
- Request Type: `GenerateReportDraftRequest`
- Response Type: `GenerateReportDraftResponse`
- 用途: 将患者上下文、抓拍图、视频帧分割信息和医生初步意见发送给 Agent 服务，返回结构化报告草稿。

### 2. 保存报告草稿
- URL: `${VITE_API_BASE_URL}/v1/report-drafts`
- Method: `POST`
- Request Type: `SaveReportDraftRequest`
- Response Type: `ReportDraftRecord`
- 用途: 持久化医生确认后的报告草稿。

### 3. 智能标签推断
- URL: `${VITE_AGENT_API_BASE_URL}/v1/annotation-tags/infer`
- Method: `POST`
- Request Type: `FetchAnnotationTagsRequest`
- Response Type: `AnnotationTag[]`
- 用途: 结合视频帧和报告片段生成病灶特征与定位标签。

## 组件规划

### PatientInfoCard
- Props: `patientName`, `gender`, `age`, `patientId`, `examDate`, `status`
- Emits: `edit(patientId)`, `view-history()`

### EndoVideoPlayer
- Props: `videoSrc`, `isPlaying`, `maskData`, `showMask`
- Emits: `play-state-change(payload)`, `capture-frame(payload)`, `update:showMask(showMask)`

### SmartAnnotationTags
- Props: `videoFrameData`, `reportSnippet`, `tags`, `isLoading`, `errorMessage`
- Emits: `fetch-agent-tags(payload)`, `tag-click(tag)`

### TumorMaskViewer
- Props: `tumorImageSrc`, `maskData`, `details`
- Emits: `toggle-mask(isVisible)`, `expand-view()`

### ReportBuilderView
- Props: `reportId`, `contextData`
- Emits: `invoke-agent(request)`, `save-draft(request)`

## MySQL 表结构建议

### patients
- `patient_id` varchar(64) primary key
- `patient_name` varchar(64)
- `gender` varchar(8)
- `age` int

### examinations
- `exam_id` bigint primary key
- `patient_id` varchar(64) indexed
- `exam_date` date
- `status` tinyint
- `scope_session_id` varchar(64)
- `report_snippet` text

### report_drafts
- `report_id` varchar(64) primary key
- `patient_id` varchar(64) indexed
- `exam_id` bigint indexed
- `findings` longtext
- `conclusion` text
- `layout_suggestion` text
- `updated_at` datetime

### report_capture_assets
- `asset_id` bigint primary key
- `report_id` varchar(64) indexed
- `asset_type` varchar(32)
- `asset_url` varchar(255)
- `captured_at_seconds` decimal(8,3)

### annotation_tags
- `tag_id` varchar(64) primary key
- `report_id` varchar(64) indexed
- `label` varchar(64)
- `confidence` decimal(4,3)
- `target_time` decimal(8,3)
- `location_label` varchar(128)
- `needs_review` tinyint

### tumor_rois
- `roi_id` varchar(64) primary key
- `report_id` varchar(64) indexed
- `image_url` varchar(255)
- `mask_payload` longtext
- `estimated_size_mm` decimal(6,2)
- `classification` varchar(64)
- `location` varchar(128)
- `surface_pattern` varchar(128)
- `confidence` decimal(4,3)

## FastAPI 后端开发清单

### 路由
- `POST /v1/report-drafts/generate`
- `POST /v1/report-drafts`
- `POST /v1/annotation-tags/infer`
- `GET /v1/patients/{patient_id}`

### Pydantic 模型
- `PatientRecordSchema`
- `PolygonMaskSchema`
- `VideoFrameDataSchema`
- `TumorDetailsSchema`
- `ReportContextDataSchema`
- `GenerateReportDraftRequestSchema`
- `GenerateReportDraftResponseSchema`
- `SaveReportDraftRequestSchema`
- `AnnotationTagSchema`

### SQLAlchemy 模型
- `Patient`
- `Examination`
- `ReportDraft`
- `ReportCaptureAsset`
- `AnnotationTag`
- `TumorRoi`