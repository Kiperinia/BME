"""兼容入口。

当前直接复用基础验证脚本，保留该文件作为历史命令入口别名。
"""

from val_base import main


if __name__ == "__main__":
    raise SystemExit(main())