from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from core.agent import build_minimal_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行最小可用的 HelloAgent 医学诊断示例")
    parser.add_argument("--image", type=Path, help="输入内镜图像路径")
    parser.add_argument("--mask", type=Path, help="输入病灶掩码路径")
    parser.add_argument("--patient-id", default="demo-patient")
    parser.add_argument("--study-id", default="demo-study")
    parser.add_argument("--exam-date", default="")
    parser.add_argument("--lesion-id", default="lesion-1")
    parser.add_argument("--pixel-size-mm", type=float, default=0.15)
    parser.add_argument("--use-llm", action="store_true", help="启用 LLM 增强推理")
    parser.add_argument("--use-llm-report", action="store_true", help="启用 LLM 报告生成")
    return parser


def load_case(image_path: Path | None, mask_path: Path | None) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    if bool(image_path) != bool(mask_path):
        raise ValueError("--image 和 --mask 必须同时提供。")

    if image_path and mask_path:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        return image, mask, None
    """如果没有提供图像和掩码，则使用默认的示例数据。"""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (28, 55, 92)
    cv2.circle(image, (128, 128), 44, (18, 42, 210), -1)
    cv2.ellipse(image, (128, 128), (52, 30), 15, 0, 360, (10, 20, 235), 4)

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 44, 255, -1)
    return image, mask, (84, 84, 172, 172)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image, mask, bbox = load_case(args.image, args.mask)
    agent = build_minimal_agent(
        use_llm=args.use_llm,
        pixel_size_mm=args.pixel_size_mm,
        use_llm_report=args.use_llm_report,
    )

    result = agent.diagnose_single_sync(
        image=image,
        mask=mask,
        bbox=bbox,
        lesion_id=args.lesion_id,
        context={
            "patient_id": args.patient_id,
            "study_id": args.study_id,
            "exam_date": args.exam_date,
        },
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()