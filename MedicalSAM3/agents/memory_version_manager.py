"""Version snapshots and rollback for exemplar memory banks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank


class MemoryVersionManager:
    """范例记忆库版本快照与回滚管理器。

    支持列出所有版本、保存新版本和回滚到指定版本。
    """
    def __init__(self, root_dir: str | Path) -> None:
        """初始化版本管理器。

        参数：
            - root_dir: 版本存储根目录

        返回：
            - None
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> list[str]:
        """列出所有可用的记忆库版本号。

        参数：
            - 无

        返回：
            - 版本号字符串排序列表
        """
        return sorted(path.stem.replace("memory_", "") for path in self.root_dir.glob("memory_v*.json"))

    def save_new_version(self, memory_bank: ExemplarMemoryBank) -> Path:
        """保存当前记忆库为新版本。

        参数：
            - memory_bank: 范例记忆库对象

        返回：
            - 保存文件的 Path 对象
        """
        return memory_bank.save(self.root_dir)

    def rollback(self, version: str) -> ExemplarMemoryBank:
        """回滚到指定版本的记忆库。

        加载目标版本并添加回滚记录到变更日志后重新保存。

        参数：
            - version: 目标版本号字符串

        返回：
            - 加载并更新后的 ExemplarMemoryBank 实例
        """
        target = self.root_dir / f"memory_{version}.json"
        if not target.exists():
            raise FileNotFoundError(f"Memory version not found: {version}")
        bank = ExemplarMemoryBank.load(target)
        bank.changelog.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "rollback",
                "from_version": bank.version,
                "requested_version": version,
            }
        )
        bank.save(self.root_dir)
        return bank
