"""带版本管理的人工校验示例记忆库。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ExemplarItem:
    """示例记忆条目数据类。

    描述单个人工校验示例的完整元信息，包括来源、路径、嵌入、
    质量评分及版本等字段，是记忆库中存储与检索的基本单元。
    """
    item_id: str
    image_id: str
    crop_path: str
    mask_path: Optional[str]
    bbox: list[float]
    embedding_path: Optional[str]
    type: str
    source_dataset: str
    fold_id: Optional[int]
    human_verified: bool
    quality_score: float
    boundary_score: float
    diversity_score: float
    difficulty_score: float
    uncertainty_score: float
    false_positive_risk: float
    created_at: str
    version: str
    notes: str


class ExemplarMemoryBank:
    """带版本管理的人工校验示例记忆库。

    维护示例条目的增删改查、版本与变更日志，并对外部数据集泄漏进行校验。

    参数：
        - items: 初始示例条目列表，缺省为空列表。
    """
    def __init__(self, items: Optional[list[ExemplarItem]] = None) -> None:
        """初始化示例记忆库。

        参数：
            - items: 初始示例条目列表。

        返回：
            - 无返回值，仅完成对象初始化。
        """
        self.items: list[ExemplarItem] = items or []
        self.rejected_items: list[dict[str, object]] = []
        self.changelog: list[dict[str, object]] = []
        self.version = "v0"

    @property
    def trainable_items(self) -> list[ExemplarItem]:
        """返回所有通过人工校验、可用于训练的示例条目。

        参数：
            - 无。

        返回：
            - 人工校验通过的示例条目列表。
        """
        return [item for item in self.items if item.human_verified]

    @classmethod
    def load(cls, path: str | Path) -> "ExemplarMemoryBank":
        """从文件或目录加载示例记忆库。

        若传入目录，则自动选取其中版本号最新的 memory_v*.json 文件；
        若路径不存在则返回空记忆库。

        参数：
            - path: JSON 文件路径或目录路径。

        返回：
            - 加载得到的 ExemplarMemoryBank 实例。
        """
        target = Path(path)
        if target.is_dir():
            candidates = sorted(target.glob("memory_v*.json"))
            if not candidates:
                return cls()
            target = candidates[-1]
        if not target.exists():
            return cls()

        payload = json.loads(target.read_text(encoding="utf-8"))
        items = [ExemplarItem(**item) for item in payload.get("items", [])]
        bank = cls(items=items)
        bank.version = payload.get("version", target.stem.replace("memory_", ""))

        rejected_path = target.parent / "rejected_items.json"
        changelog_path = target.parent / "changelog.json"
        if rejected_path.exists():
            bank.rejected_items = json.loads(rejected_path.read_text(encoding="utf-8"))
        if changelog_path.exists():
            bank.changelog = json.loads(changelog_path.read_text(encoding="utf-8"))
        return bank

    def _record_change(self, action: str, item_id: str, details: Optional[dict[str, object]] = None) -> None:
        """在变更日志中追加一条操作记录。

        参数：
            - action: 操作类型（如 add、remove、reject）。
            - item_id: 被操作的示例条目 ID。
            - details: 附加详情字典，缺省为空。

        返回：
            - 无返回值，仅更新内存中的变更日志。
        """
        self.changelog.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.version,
                "action": action,
                "item_id": item_id,
                "details": details or {},
            }
        )

    def _validate_item(self, item: ExemplarItem, external_dataset_names: Optional[list[str]] = None) -> None:
        """校验示例条目是否包含外部数据集泄漏。

        参数：
            - item: 待校验的示例条目。
            - external_dataset_names: 不允许出现的外部数据集名称列表，缺省为 ["PolypGen"]。

        返回：
            - 无返回值；若检测到泄漏则抛出 ValueError。
        """
        external_dataset_names = external_dataset_names or ["PolypGen"]
        if any(name.lower() in item.source_dataset.lower() for name in external_dataset_names):
            raise ValueError(f"External dataset leakage detected for item {item.item_id}: {item.source_dataset}")

    def add_item(self, item: ExemplarItem) -> None:
        """向记忆库中添加示例条目。

        若已存在相同 item_id 的条目则先移除再添加，并在变更日志中记录。

        参数：
            - item: 待添加的示例条目。

        返回：
            - 无返回值，仅更新内存中的条目列表与日志。
        """
        self._validate_item(item)
        self.items = [existing for existing in self.items if existing.item_id != item.item_id]
        self.items.append(item)
        self._record_change("add", item.item_id, {"human_verified": item.human_verified, "type": item.type})

    def remove_item(self, item_id: str) -> None:
        """从记忆库中移除指定 ID 的示例条目。

        参数：
            - item_id: 待移除条目的 ID。

        返回：
            - 无返回值，仅更新内存中的条目列表与日志。
        """
        self.items = [item for item in self.items if item.item_id != item_id]
        self._record_change("remove", item_id)

    def get_items(
        self,
        type: Optional[str] = None,
        source_dataset: Optional[str] = None,
        human_verified: Optional[bool] = None,
    ) -> list[ExemplarItem]:
        """按条件筛选示例条目。

        参数：
            - type: 按示例类型筛选（如 positive、negative、boundary）；为 None 则不筛选。
            - source_dataset: 按来源数据集名称筛选；为 None 则不筛选。
            - human_verified: 按是否人工校验筛选；为 None 则不筛选。

        返回：
            - 满足全部筛选条件的示例条目列表。
        """
        items = self.items
        if type is not None:
            items = [item for item in items if item.type == type]
        if source_dataset is not None:
            items = [item for item in items if item.source_dataset == source_dataset]
        if human_verified is not None:
            items = [item for item in items if item.human_verified == human_verified]
        return items

    def reject_item(self, item_id: str, reason: str) -> None:
        """将指定 ID 的条目移入拒绝列表并记录拒绝原因。

        参数：
            - item_id: 待拒绝条目的 ID。
            - reason: 拒绝原因描述。

        返回：
            - 无返回值，仅更新内存中的条目列表、拒绝列表与日志。
        """
        matched = None
        remaining = []
        for item in self.items:
            if item.item_id == item_id:
                matched = item
            else:
                remaining.append(item)
        self.items = remaining
        record = {
            "item_id": item_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item": asdict(matched) if matched is not None else None,
        }
        self.rejected_items.append(record)
        self._record_change("reject", item_id, {"reason": reason})

    def _next_version_path(self, directory: Path) -> Path:
        """在给定目录下生成下一个版本的记忆库文件路径。

        同时将记忆库版本号更新为新版本，并为所有条目打上该版本标记。

        参数：
            - directory: 目标目录路径。

        返回：
            - 新版本对应的 JSON 文件路径。
        """
        directory.mkdir(parents=True, exist_ok=True)
        version_index = 0
        while (directory / f"memory_v{version_index}.json").exists():
            version_index += 1
        self.version = f"v{version_index}"
        for item in self.items:
            item.version = self.version
        return directory / f"memory_{self.version}.json"

    def save(self, path: str | Path) -> Path:
        """将记忆库持久化到磁盘。

        若传入 .json 文件路径则直接写入；若传入目录则自动生成下一个版本文件。
        同时将拒绝列表与变更日志写入同目录下的配套文件。

        参数：
            - path: 目标文件或目录路径。

        返回：
            - 实际写入的 JSON 文件路径。
        """
        destination = Path(path)
        if destination.suffix == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.stem.startswith("memory_v"):
                self.version = destination.stem.replace("memory_", "")
        else:
            destination = self._next_version_path(destination)
        payload = {
            "version": self.version,
            "items": [asdict(item) for item in self.items],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / "rejected_items.json").write_text(
            json.dumps(self.rejected_items, indent=2), encoding="utf-8"
        )
        (destination.parent / "changelog.json").write_text(
            json.dumps(self.changelog, indent=2), encoding="utf-8"
        )
        return destination

    def export_changelog(self, path: str | Path) -> Path:
        """将变更日志导出为独立的 JSON 文件。

        参数：
            - path: 导出文件路径。

        返回：
            - 实际写入的文件路径。
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.changelog, indent=2), encoding="utf-8")
        return destination

    def check_no_external_leakage(self, external_dataset_names: list[str] = ["PolypGen"]) -> bool:
        """检查记忆库中是否存在外部数据集泄漏。

        参数：
            - external_dataset_names: 不允许出现的外部数据集名称列表，缺省为 ["PolypGen"]。

        返回：
            - 若不存在任何外部数据集条目返回 True，否则返回 False。
        """
        return not any(
            any(dataset.lower() in item.source_dataset.lower() for dataset in external_dataset_names)
            for item in self.items
        )
