"""为 MedEx-SAM3 分割文件生成 YOLO 边界框缓存。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.scripts.common import read_records
from MedicalSAM3.yolo_adapter.detector import UltralyticsYoloDetector


def main() -> int:
    """命令行入口，根据分割记录文件生成 YOLO 边界框缓存。

    解析命令行参数，读取分割记录，对每张图像运行 YOLO 检测，
    将结果写入 JSON 缓存文件，并记录缺失和错误项。

    返回：
        - 退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="Generate YOLO bbox cache from MedEx-SAM3 split records.")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="yolo/models/yolov8_polyp.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    records = read_records(args.split_file)
    detector = UltralyticsYoloDetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )

    cache = {}
    missing = []
    errors = []
    for index, record in enumerate(records, start=1):
        image_path = str(record.get("image_path", ""))
        key = str(record.get("image_id") or Path(image_path).stem)
        try:
            detection = detector.predict_one(image_path) if image_path and Path(image_path).is_file() else None
        except Exception as exc:
            detection = None
            errors.append({"image_id": key, "image_path": image_path, "error": str(exc)})
        if detection is None:
            missing.append(key)
            continue
        cache[key] = detection.to_dict()
        if index == 1 or index % 50 == 0 or index == len(records):
            print(json.dumps({"progress": index, "total": len(records), "cached": len(cache), "missing": len(missing)}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    (output.parent / f"{output.stem}_missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    (output.parent / f"{output.stem}_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cached": len(cache), "missing": len(missing), "errors": len(errors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
