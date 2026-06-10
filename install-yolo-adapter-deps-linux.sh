#!/usr/bin/env bash
set -euo pipefail

# Install only the packages needed by the YOLO bbox adapter.
# This script intentionally does not install/upgrade torch so it will not
# disturb an existing CUDA-specific PyTorch environment.

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[yolo-adapter] Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[yolo-adapter] Upgrading pip tooling"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[yolo-adapter] Installing YOLO adapter runtime dependencies"
"$PYTHON_BIN" -m pip install \
  "ultralytics>=8.0.0" \
  "opencv-python>=4.8.0" \
  "pillow>=10.0.0" \
  "numpy>=1.24.0" \
  "pyyaml>=6.0.0"

echo "[yolo-adapter] Verifying imports"
"$PYTHON_BIN" - <<'PY'
import importlib

for name in ["ultralytics", "cv2", "PIL", "numpy", "yaml"]:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
PY

echo "[yolo-adapter] Done"
