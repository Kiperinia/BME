from __future__ import annotations

from app.core.exceptions import AppException
from app.schemas.agent_workflow import (
    PatientContextSchema,
    PolygonMaskSchema,
    ReportContextSchema,
    TumorDetailsSchema,
    TumorFocusSchema,
    VideoFrameDataSchema,
)


PATIENT_PREVIEW_FIXTURES = [
    {
        "patientId": "EIS-2026-000128",
        "patientName": "张明远",
        "gender": "男",
        "age": 58,
        "examDate": "2026-04-16",
        "status": 1,
    },
    {
        "patientId": "EIS-2026-000129",
        "patientName": "李若兰",
        "gender": "女",
        "age": 46,
        "examDate": "2026-04-15",
        "status": 2,
    },
    {
        "patientId": "EIS-2026-000130",
        "patientName": "周航",
        "gender": "男",
        "age": 63,
        "examDate": "2026-04-16",
        "status": 0,
    },
]


class ReportContextService:
    def list_patient_previews(self) -> list[PatientContextSchema]:
        return [PatientContextSchema(**record) for record in PATIENT_PREVIEW_FIXTURES]

    def get_report_context(
        self,
        report_id: str | None = None,
        patient_id: str | None = None,
    ) -> ReportContextSchema:
        patient = self._resolve_patient(patient_id=patient_id)
        frame_id_seed = report_id or patient.patientId
        source_id = report_id or "scope-session-20260416-01"

        default_mask = PolygonMaskSchema(
            id="mask-frame-1",
            frameWidth=1024,
            frameHeight=1024,
            fillColor="rgba(37, 99, 235, 0.26)",
            strokeColor="rgba(37, 99, 235, 0.9)",
            points=[
                (372, 284),
                (458, 244),
                (582, 252),
                (658, 346),
                (648, 482),
                (534, 568),
                (396, 538),
                (332, 420),
            ],
        )
        tumor_mask = PolygonMaskSchema(
            id="tumor-roi-1",
            frameWidth=1200,
            frameHeight=900,
            fillColor="rgba(16, 185, 129, 0.28)",
            strokeColor="rgba(5, 150, 105, 0.95)",
            points=[
                (364, 266),
                (520, 216),
                (704, 274),
                (748, 436),
                (640, 584),
                (438, 612),
                (320, 480),
            ],
        )

        return ReportContextSchema(
            patient=patient,
            videoSrc="",
            maskData=[default_mask],
            showMask=True,
            videoFrameData=VideoFrameDataSchema(
                frameId=f"{frame_id_seed}-frame-001",
                sourceId=source_id,
                timestamp=12.5,
                width=1024,
                height=1024,
                suspectedLocation="乙状结肠",
            ),
            captureImageSrcs=["/images/endoscopy-frame-demo.svg", "/images/tumor-roi-demo.svg"],
            reportSnippet="乙状结肠见一枚约 6 mm 隆起性病灶，边界尚清，建议结合病理。",
            initialOpinion="请基于抓拍图、视频分割结果和病灶部位，生成符合 EIS 规范的内镜报告草稿。",
            tumorFocus=TumorFocusSchema(
                tumorImageSrc="/images/tumor-roi-demo.svg",
                maskData=[tumor_mask],
                details=TumorDetailsSchema(
                    estimatedSizeMm=6.4,
                    classification="疑似管状腺瘤",
                    location="乙状结肠距肛缘约 28 cm",
                    surfacePattern="表面细颗粒样，边缘轻度隆起",
                    confidence=0.92,
                ),
            ),
        )

    def _resolve_patient(self, patient_id: str | None) -> PatientContextSchema:
        if not patient_id:
            return PatientContextSchema(**PATIENT_PREVIEW_FIXTURES[0])

        matched = next((record for record in PATIENT_PREVIEW_FIXTURES if record["patientId"] == patient_id), None)
        if not matched:
            raise AppException(404, 40421, f"patient {patient_id} was not found")

        return PatientContextSchema(**matched)