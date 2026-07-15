#!/usr/bin/env bash
set -euo pipefail

# Install only the packages needed by the YOLO bbox adapter.
# This script intentionally does not install/upgrade torch so it will not
# disturb an existing CUDA-specific PyTorch environment.

USE_CONDA="${USE_CONDA:-1}"
CONDA_ENV="${CONDA_ENV:-sam3wangruifeng}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python() {
  if [ "$USE_CONDA" = "1" ]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

if [ "$USE_CONDA" = "1" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found. Set USE_CONDA=0 only when a local Python is explicitly intended." >&2
    exit 1
  fi
  if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "[error] conda environment not found: $CONDA_ENV" >&2
    exit 1
  fi
fi

echo "[yolo-adapter] Conda mode: $USE_CONDA"
echo "[yolo-adapter] Conda env: $CONDA_ENV"
echo "[yolo-adapter] Python: $(run_python -c 'import sys; print(sys.executable)')"
echo "[yolo-adapter] Upgrading pip tooling"
run_python -m pip install --upgrade pip setuptools wheel

echo "[yolo-adapter] Installing YOLO adapter runtime dependencies"
run_python -m pip install \
  "ultralytics>=8.0.0" \
  "opencv-python>=4.8.0" \
  "pillow>=10.0.0" \
  "numpy>=1.24.0" \
  "pyyaml>=6.0.0"

echo "[yolo-adapter] Verifying imports"
run_python - <<'PY'
import importlib

for name in ["ultralytics", "cv2", "PIL", "numpy", "yaml"]:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
PY

echo "[yolo-adapter] Done"
