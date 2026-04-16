# Async SAM3 Worker Backend

## 模块概览

- FastAPI 作为生产者，负责接收图片、写入任务记录、派发 Celery 任务。
- Redis 同时承担 Broker 与 Result Backend。
- Celery Worker 使用单例方式预热 SAM3 运行时，避免重复加载模型占用显存。
- SQLAlchemy Async 负责持久化任务状态和病灶结果。

## 目录结构

```text
Backend/
  app/
    api/
    core/
    models/
    repositories/
    schemas/
    services/
    worker/
  .env.example
  requirements.txt
```

## 环境变量

1. 复制 .env.example 为 .env。
2. 至少配置 MYSQL_URL、CELERY_BROKER_URL、CELERY_RESULT_BACKEND。
3. Windows 下建议保留 CELERY_WORKER_POOL=solo 且 CELERY_WORKER_CONCURRENCY=1。

## 启动命令

```powershell
cd E:\BME\Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
cd E:\BME\Backend
celery -A app.worker.celery_app:celery_app worker --loglevel=info --pool=solo --concurrency=1
```

## 接口契约

### POST /api/analysis/segment-frame

- Content-Type: multipart/form-data
- Fields:
  - image: 单帧内镜图像或病灶截图
- 行为:
  - 路由通过依赖注入调用 SAM3Engine 单例，避免重复加载权重。
  - 默认根据 MODEL_LOAD_MODE 在 mock 与真实 SAM3 模式之间切换。
  - 同步接口受 MODEL_INFERENCE_TIMEOUT_SECONDS 控制，超时返回 504。

成功响应示例:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "mask_coordinates": [[124, 88], [188, 92], [201, 146], [136, 152]],
    "bounding_box": [118, 88, 201, 152]
  }
}
```

说明:

- mock 模式下直接返回硬编码测试多边形，保证 EndoVideoPlayer 与 TumorMaskViewer 的前端联调不中断。
- sam3 模式下，后端会预热单例模型，并将二维 mask 通过 OpenCV 轮廓提取与 Douglas-Peucker 近似转成前端可绘制多边形。
- 当前仓库中的 MedicalSAM3 包装器需要 prompt 才能稳定推理，因此同步接口默认使用全图先验框作为无外部 prompt 的兜底策略；后续若接入自动 prompt 或 LoRA 扩展，不需要修改 API 契约。

### POST /api/analysis/submit-task

- Content-Type: multipart/form-data
- Fields:
  - image: 图片文件
  - patient_id: 患者编号
  - study_id: 检查编号，可选
  - lesion_hint: 病灶提示，可选

成功响应示例:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "714ce71b-1d25-4a37-b40e-8aa55d4f9744",
    "status": "PENDING"
  }
}
```

### GET /api/analysis/task-status/{task_id}

成功响应示例:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "714ce71b-1d25-4a37-b40e-8aa55d4f9744",
    "status": "SUCCESS",
    "patient_id": "PATIENT_001",
    "study_id": "EXAM_20260416_001",
    "submitted_at": "2026-04-16T13:40:00+08:00",
    "started_at": "2026-04-16T13:40:01+08:00",
    "completed_at": "2026-04-16T13:40:04+08:00",
    "mask_coordinates": [
      [
        {"x": 124, "y": 88},
        {"x": 188, "y": 92},
        {"x": 201, "y": 146},
        {"x": 136, "y": 152}
      ]
    ],
    "lesions": [
      {
        "lesion_id": "6a8c5af0-5805-4809-b29c-a0b39ca02fb5",
        "label": "suspected_polyp",
        "confidence": 0.973,
        "location": "sigmoid_colon",
        "area_mm2": 18.4,
        "mask_coordinates": [
          {"x": 124, "y": 88},
          {"x": 188, "y": 92},
          {"x": 201, "y": 146},
          {"x": 136, "y": 152}
        ]
      }
    ],
    "error_code": null,
    "error_message": null
  }
}
```

## curl 联调示例

```powershell
curl -X POST "http://127.0.0.1:8000/api/analysis/submit-task" ^
  -H "X-User-Id: demo-doctor" ^
  -F "patient_id=PATIENT_001" ^
  -F "study_id=EXAM_20260416_001" ^
  -F "lesion_hint=suspected sigmoid lesion" ^
  -F "image=@E:/BME/sample/endoscopy.png"
```

```powershell
curl "http://127.0.0.1:8000/api/analysis/task-status/714ce71b-1d25-4a37-b40e-8aa55d4f9744" ^
  -H "X-User-Id: demo-doctor"
```