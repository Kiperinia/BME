from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException
from app.models.patient import Patient
from app.repositories.patient_repository import PatientRepository
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

WORKSPACE_DIR = Path(__file__).resolve().parents[3]
MEDICALSAM3_OUTPUTS_DIR = WORKSPACE_DIR / "MedicalSAM3" / "outputs"
MEDEX_VISUALIZATION_DIR = MEDICALSAM3_OUTPUTS_DIR / "medex_sam3" / "eval_local_mask_try" / "visualizations"
MEDEX_ASSET_PREFIX = "/api/assets/medex-sam3"
logger = logging.getLogger(__name__)


class ReportContextService:
    def __init__(self, session: AsyncSession | None = None):
        self.session = session
        self.repository = None if session is None else PatientRepository(session=session)

    async def list_patient_previews(self) -> list[PatientContextSchema]:
        patients = await self._list_patients()
        return [self._patient_to_schema(patient) for patient in patients]

    async def get_report_context(
        self,
        report_id: str | None = None,
        patient_id: str | None = None,
    ) -> ReportContextSchema:
        patient = await self._resolve_patient(patient_id=patient_id)
        frame_id_seed = report_id or patient.patientId
        source_id = report_id or "scope-session-20260416-01"
        visualization_assets = self._resolve_medex_visualization_assets()

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
            captureImageSrcs=visualization_assets["captureImageSrcs"] if visualization_assets else ["/images/endoscopy-frame-demo.svg", "/images/tumor-roi-demo.svg"],
            reportSnippet="乙状结肠见一枚约 6 mm 隆起性病灶，边界尚清，建议结合病理。",
            initialOpinion="请基于抓拍图、视频分割结果和病灶部位，生成符合 EIS 规范的内镜报告草稿。",
            tumorFocus=TumorFocusSchema(
                tumorImageSrc=visualization_assets["tumorImageSrc"] if visualization_assets else "/images/tumor-roi-demo.svg",
                maskData=visualization_assets["tumorMaskSrc"] if visualization_assets else [tumor_mask],
                details=TumorDetailsSchema(
                    estimatedSizeMm=6.4,
                    classification="疑似管状腺瘤",
                    location="乙状结肠距肛缘约 28 cm",
                    surfacePattern="表面细颗粒样，边缘轻度隆起",
                    confidence=0.92,
                ),
            ),
        )

    async def ensure_seed_data(self) -> None:
        if self.repository is None:
            return

        seed_rows = [
            Patient(
                patient_id=record["patientId"],
                patient_name=record["patientName"],
                gender=record["gender"],
                age=record["age"],
                exam_date=record["examDate"],
                status=record["status"],
            )
            for record in PATIENT_PREVIEW_FIXTURES
        ]
        await self.repository.upsert_many(seed_rows)

    async def _resolve_patient(self, patient_id: str | None) -> PatientContextSchema:
        patients = await self._list_patients()
        if not patients:
            raise AppException(404, 40421, "no patients were found")

        if not patient_id:
            return self._patient_to_schema(patients[0])

        matched = next((patient for patient in patients if patient.patient_id == patient_id), None)
        if matched is None:
            raise AppException(404, 40421, f"patient {patient_id} was not found")

        return self._patient_to_schema(matched)

    async def _list_patients(self) -> list[Patient]:
        if self.repository is not None:
            try:
                patients = await self.repository.list_all()
                if patients:
                    return patients
            except Exception as exc:
                logger.warning("Falling back to in-memory patient fixtures because database lookup failed: %s", exc)

        return [
            Patient(
                patient_id=record["patientId"],
                patient_name=record["patientName"],
                gender=record["gender"],
                age=record["age"],
                exam_date=record["examDate"],
                status=record["status"],
            )
            for record in PATIENT_PREVIEW_FIXTURES
        ]

    @staticmethod
    def _patient_to_schema(patient: Patient) -> PatientContextSchema:
        return PatientContextSchema(
            patientId=patient.patient_id,
            patientName=patient.patient_name,
            gender=patient.gender,
            age=patient.age,
            examDate=patient.exam_date,
            status=patient.status,
        )

    def _resolve_medex_visualization_assets(self) -> dict[str, str | list[str]] | None:
        if not MEDEX_VISUALIZATION_DIR.exists():
            return None

        image_files = sorted(MEDEX_VISUALIZATION_DIR.glob("*_image.png"))
        if not image_files:
            return None

        base_name = image_files[0].name.removesuffix("_image.png")
        image_path = MEDEX_VISUALIZATION_DIR / f"{base_name}_image.png"
        pred_path = MEDEX_VISUALIZATION_DIR / f"{base_name}_pred.png"
        overlay_path = MEDEX_VISUALIZATION_DIR / f"{base_name}_boundary_overlay.png"
        gt_path = MEDEX_VISUALIZATION_DIR / f"{base_name}_gt.png"

        required_paths = [image_path, pred_path, overlay_path]
        if any(not path.exists() for path in required_paths):
            return None

        capture_paths = [overlay_path, pred_path]
        if gt_path.exists():
            capture_paths.append(gt_path)
        capture_paths.append(image_path)

        return {
            "tumorImageSrc": self._to_medex_asset_url(image_path),
            "tumorMaskSrc": self._to_medex_asset_url(pred_path),
            "captureImageSrcs": [self._to_medex_asset_url(path) for path in capture_paths],
        }

    @staticmethod
    def _to_medex_asset_url(path: Path) -> str:
        relative_path = path.relative_to(MEDICALSAM3_OUTPUTS_DIR).as_posix()
        return f"{MEDEX_ASSET_PREFIX}/{relative_path}"