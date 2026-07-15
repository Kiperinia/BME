"""使用逐图像 delta-Dice 验证反馈策展示例库。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ROLE_TO_TYPE = {
    "positive_ids": "positive",
    "negative_ids": "negative",
    "boundary_ids": "boundary",
}


@dataclass
class UsageStats:
    """记录每个示例项在验证中的使用统计信息。

    参数：
        - 无

    返回：
        - 提供用于策展决策的统计实例
    """
    used: int = 0
    positive_delta_count: int = 0
    negative_delta_count: int = 0
    severe_positive_count: int = 0
    severe_negative_count: int = 0
    sum_delta: float = 0.0
    min_delta: float = 0.0
    max_delta: float = 0.0

    def update(self, delta: float, severe_delta: float) -> None:
        """更新统计信息，累加一次使用记录。

        参数：
            - delta: 本次使用的 Dice 增量
            - severe_delta: 被认为是"严重"变化的阈值

        返回：
            - 无返回值，仅更新内部计数器与汇总值
        """
        if self.used == 0:
            self.min_delta = delta
            self.max_delta = delta
        else:
            self.min_delta = min(self.min_delta, delta)
            self.max_delta = max(self.max_delta, delta)
        self.used += 1
        self.sum_delta += delta
        if delta > 0:
            self.positive_delta_count += 1
        elif delta < 0:
            self.negative_delta_count += 1
        if delta >= severe_delta:
            self.severe_positive_count += 1
        elif delta <= -severe_delta:
            self.severe_negative_count += 1

    @property
    def mean_delta(self) -> float:
        """计算平均 Dice 增量。

        参数：
            - 无

        返回：
            - 累加增量总和除以使用次数
        """
        return self.sum_delta / max(self.used, 1)

    def as_dict(self) -> dict[str, Any]:
        """将统计信息序列化为字典。

        参数：
            - 无

        返回：
            - 包含所有统计字段的字典
        """
        return {
            "used": self.used,
            "positive_delta_count": self.positive_delta_count,
            "negative_delta_count": self.negative_delta_count,
            "severe_positive_count": self.severe_positive_count,
            "severe_negative_count": self.severe_negative_count,
            "sum_delta": self.sum_delta,
            "mean_delta": self.mean_delta,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
        }


def _load_bank(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """从文件或目录加载记忆库。

    参数：
        - path: 记忆库文件路径或目录路径（自动查找最新版本）

    返回：
        - 由 (库文件路径, 库数据字典) 组成的元组
    """
    target = Path(path)
    if target.is_dir():
        candidates = sorted(target.glob("memory_v*.json"))
        if not candidates:
            raise FileNotFoundError(f"No memory_v*.json found in {target}")
        target = candidates[-1]
    if not target.exists():
        raise FileNotFoundError(f"Memory bank not found: {target}")
    return target, json.loads(target.read_text(encoding="utf-8"))


def _load_per_image_metrics(path: str | Path) -> list[dict[str, Any]]:
    """从 JSON 或 JSONL 文件加载逐图像指标。

    参数：
        - path: 指标文件路径

    返回：
        - 指标字典列表
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Per-image metrics not found: {target}")
    if target.suffix == ".jsonl":
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("per_image_metrics must be a JSON array or JSONL rows")
    return payload


def _collect_usage_stats(rows: list[dict[str, Any]], severe_delta: float) -> dict[str, UsageStats]:
    """从逐图像指标中收集每个示例项的使用统计。

    参数：
        - rows: 逐图像指标行列表
        - severe_delta: 严重变化阈值

    返回：
        - 示例 ID 到 UsageStats 的映射字典
    """
    stats: dict[str, UsageStats] = defaultdict(UsageStats)
    for row in rows:
        delta = float(row.get("delta_dice", 0.0))
        selected = row.get("selected_exemplars") or {}
        for role in ROLE_TO_TYPE:
            for item_id in selected.get(role, []) or []:
                stats[item_id].update(delta, severe_delta)
    return stats


def _score_item(item: dict[str, Any], stats: UsageStats | None) -> float:
    """综合基础质量评分与使用统计信息计算示例项的综合得分。

    参数：
        - item: 示例项字典
        - stats: 使用统计，为 None 时不加统计项

    返回：
        - 综合得分浮点值
    """
    base_score = (
        float(item.get("quality_score", 0.0))
        + 0.5 * float(item.get("boundary_score", 0.0))
        + 0.5 * float(item.get("diversity_score", 0.0))
        - 0.5 * float(item.get("uncertainty_score", 0.0))
        - 0.25 * float(item.get("false_positive_risk", 0.0))
    )
    if stats is None or stats.used == 0:
        return base_score
    return base_score + 10.0 * stats.mean_delta + 0.05 * stats.positive_delta_count - 0.05 * stats.negative_delta_count


def _is_bad_item(
    stats: UsageStats | None,
    *,
    min_used: int,
    max_bad_mean_delta: float,
    severe_negative_delta: float,
    min_severe_negative_count: int,
    negative_majority_margin: int,
) -> tuple[bool, str]:
    """判断示例项是否表现不佳（平均增量差或严重负面使用过多）。

    参数：
        - stats: 使用统计
        - min_used: 最小使用次数要求
        - max_bad_mean_delta: 被视为"差"的最大平均增量
        - severe_negative_delta: 严重负面增量阈值
        - min_severe_negative_count: 严重负面使用最小次数
        - negative_majority_margin: 负面比正面多出的最小数量

    返回：
        - 由 (是否差, 原因字符串) 组成的元组
    """
    if stats is None or stats.used < min_used:
        return False, ""
    if stats.mean_delta < max_bad_mean_delta and stats.negative_delta_count >= stats.positive_delta_count + negative_majority_margin:
        return True, (
            f"mean_delta={stats.mean_delta:.6f}, "
            f"negative_delta_count={stats.negative_delta_count}, "
            f"positive_delta_count={stats.positive_delta_count}"
        )
    if stats.severe_negative_count >= min_severe_negative_count and stats.severe_negative_count > stats.severe_positive_count:
        return True, (
            f"severe_negative_count={stats.severe_negative_count}, "
            f"severe_positive_count={stats.severe_positive_count}, "
            f"min_delta={stats.min_delta:.6f}"
        )
    return False, ""


def _protect_top_items(
    items: list[dict[str, Any]],
    stats_by_id: dict[str, UsageStats],
    min_items_per_type: int,
) -> set[str]:
    """保护每个类型中得分最高的示例项不被删除。

    参数：
        - items: 示例项列表
        - stats_by_id: 示例 ID 到使用统计的映射
        - min_items_per_type: 每种类型最少保留数量

    返回：
        - 被保护示例项的 ID 集合
    """
    protected: set[str] = set()
    for exemplar_type in ["positive", "boundary", "negative"]:
        typed = [item for item in items if item.get("type") == exemplar_type]
        typed.sort(key=lambda item: _score_item(item, stats_by_id.get(item["item_id"])), reverse=True)
        protected.update(item["item_id"] for item in typed[: max(min_items_per_type, 0)])
    return protected


def _write_curated_bank(
    *,
    source_bank_path: Path,
    source_bank: dict[str, Any],
    output_dir: Path,
    kept_items: list[dict[str, Any]],
    rejected_items: list[dict[str, Any]],
    stats_by_id: dict[str, UsageStats],
    args: argparse.Namespace,
) -> None:
    """将策展后的记忆库及报告写入输出目录。

    参数：
        - source_bank_path: 源库文件路径
        - source_bank: 源库数据字典
        - output_dir: 输出目录
        - kept_items: 保留的示例项列表
        - rejected_items: 被拒绝的示例项列表
        - stats_by_id: 使用统计映射
        - args: 命令行参数

    返回：
        - 无返回值，仅执行写入文件的副作用
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    version = args.version
    bank_path = output_dir / f"memory_{version}.json"
    bank_payload = {
        "version": version,
        "source_bank": str(source_bank_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": kept_items,
    }
    bank_path.write_text(json.dumps(bank_payload, indent=2), encoding="utf-8")

    report = {
        "source_bank": str(source_bank_path),
        "output_bank": str(bank_path),
        "source_count": len(source_bank.get("items", [])),
        "kept_count": len(kept_items),
        "rejected_count": len(rejected_items),
        "kept_by_type": {
            exemplar_type: sum(1 for item in kept_items if item.get("type") == exemplar_type)
            for exemplar_type in ["positive", "boundary", "negative"]
        },
        "rejected_by_type": {
            exemplar_type: sum(1 for item in rejected_items if item.get("type") == exemplar_type)
            for exemplar_type in ["positive", "boundary", "negative"]
        },
        "thresholds": {
            "min_used": args.min_used,
            "max_bad_mean_delta": args.max_bad_mean_delta,
            "severe_delta": args.severe_delta,
            "min_severe_negative_count": args.min_severe_negative_count,
            "negative_majority_margin": args.negative_majority_margin,
            "min_items_per_type": args.min_items_per_type,
        },
        "top_bad_usage": sorted(
            [
                {"item_id": item_id, **stats.as_dict()}
                for item_id, stats in stats_by_id.items()
            ],
            key=lambda row: (row["mean_delta"], -row["used"]),
        )[:50],
        "top_good_usage": sorted(
            [
                {"item_id": item_id, **stats.as_dict()}
                for item_id, stats in stats_by_id.items()
            ],
            key=lambda row: (row["mean_delta"], row["used"]),
            reverse=True,
        )[:50],
    }
    (output_dir / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "rejected_items.json").write_text(json.dumps(rejected_items, indent=2), encoding="utf-8")
    (output_dir / "usage_stats.json").write_text(
        json.dumps({item_id: stats.as_dict() for item_id, stats in stats_by_id.items()}, indent=2),
        encoding="utf-8",
    )

    for sidecar in ["review_queue.csv", "bank_stats.json"]:
        src = source_bank_path.parent / sidecar
        if src.exists():
            shutil.copy2(src, output_dir / sidecar)


def main() -> int:
    """脚本命令行入口，使用验证 delta-Dice 反馈策展示例库。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
    parser = argparse.ArgumentParser(description="Curate an exemplar bank using validation delta-Dice feedback.")
    parser.add_argument("--memory-bank", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--per-image-metrics", default="MedicalSAM3/outputs/medex_sam3/validation/polypgen_exemplar_delta/per_image_metrics.json")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank_curated")
    parser.add_argument("--version", default="v0")
    parser.add_argument("--min-used", type=int, default=20)
    parser.add_argument("--max-bad-mean-delta", type=float, default=-0.002)
    parser.add_argument("--severe-delta", type=float, default=0.10)
    parser.add_argument("--min-severe-negative-count", type=int, default=3)
    parser.add_argument("--negative-majority-margin", type=int, default=1)
    parser.add_argument("--min-items-per-type", type=int, default=8)
    args = parser.parse_args()

    source_bank_path, bank = _load_bank(args.memory_bank)
    rows = _load_per_image_metrics(args.per_image_metrics)
    stats_by_id = _collect_usage_stats(rows, severe_delta=args.severe_delta)
    source_items = list(bank.get("items", []))
    protected_ids = _protect_top_items(source_items, stats_by_id, args.min_items_per_type)

    kept_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    for item in source_items:
        item_id = item["item_id"]
        stats = stats_by_id.get(item_id)
        is_bad, reason = _is_bad_item(
            stats,
            min_used=args.min_used,
            max_bad_mean_delta=args.max_bad_mean_delta,
            severe_negative_delta=args.severe_delta,
            min_severe_negative_count=args.min_severe_negative_count,
            negative_majority_margin=args.negative_majority_margin,
        )
        if is_bad and item_id not in protected_ids:
            rejected_items.append(
                {
                    "item": item,
                    "reason": reason,
                    "usage_stats": stats.as_dict() if stats is not None else None,
                }
            )
            continue
        curated_item = dict(item)
        curated_item["notes"] = f"{curated_item.get('notes', '')} | curated_by_delta_feedback".strip()
        kept_items.append(curated_item)

    _write_curated_bank(
        source_bank_path=source_bank_path,
        source_bank=bank,
        output_dir=Path(args.output_dir),
        kept_items=kept_items,
        rejected_items=rejected_items,
        stats_by_id=stats_by_id,
        args=args,
    )

    print(
        json.dumps(
            {
                "source_bank": str(source_bank_path),
                "output_dir": args.output_dir,
                "source_count": len(source_items),
                "kept_count": len(kept_items),
                "rejected_count": len(rejected_items),
                "kept_by_type": {
                    exemplar_type: sum(1 for item in kept_items if item.get("type") == exemplar_type)
                    for exemplar_type in ["positive", "boundary", "negative"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
