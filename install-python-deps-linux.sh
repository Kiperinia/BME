#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_CONDA="${USE_CONDA:-1}"
CONDA_ENV="${CONDA_ENV:-sam3wangruifeng}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-linux}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu126}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements/model-requirements.txt}"
EXTRA_REQUIREMENTS="${EXTRA_REQUIREMENTS:-$ROOT_DIR/requirements/backend-requirements.txt $ROOT_DIR/requirements/agent-requirements.txt}"
RUN_IN_TMUX="${RUN_IN_TMUX:-0}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-bme-python-deps}"

start_in_tmux() {
  local command
  command=$(printf 'cd %q && RUN_IN_TMUX=0 USE_CONDA=%q CONDA_ENV=%q PYTHON_BIN=%q VENV_DIR=%q TORCH_CHANNEL=%q REQUIREMENTS_FILE=%q EXTRA_REQUIREMENTS=%q bash %q' \
    "$ROOT_DIR" "$USE_CONDA" "$CONDA_ENV" "$PYTHON_BIN" "$VENV_DIR" "$TORCH_CHANNEL" "$REQUIREMENTS_FILE" "$EXTRA_REQUIREMENTS" "$ROOT_DIR/install-python-deps-linux.sh")
  tmux new-session -d -s "$TMUX_SESSION_NAME" "$command"
  echo "[info] started tmux session: $TMUX_SESSION_NAME"
  echo "[info] attach with: tmux attach -t $TMUX_SESSION_NAME"
  exit 0
}

run_python() {
  if [ "$USE_CONDA" = "1" ]; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

if [ "$RUN_IN_TMUX" = "1" ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[error] tmux not found but RUN_IN_TMUX=1 was requested" >&2
    exit 1
  fi
  if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "[error] tmux session already exists: $TMUX_SESSION_NAME" >&2
    echo "[error] attach with: tmux attach -t $TMUX_SESSION_NAME, or set a different TMUX_SESSION_NAME" >&2
    exit 1
  fi
  start_in_tmux
fi

if [ "$USE_CONDA" = "1" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found. Set USE_CONDA=0 only when a local venv is explicitly intended." >&2
    exit 1
  fi
  if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "[error] conda environment not found: $CONDA_ENV" >&2
    echo "[error] create or activate the approved server environment before installing dependencies" >&2
    exit 1
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[error] Python not found: $PYTHON_BIN" >&2
    exit 1
  fi
fi

echo "[info] root: $ROOT_DIR"
echo "[info] use conda: $USE_CONDA"
echo "[info] conda env: $CONDA_ENV"
echo "[info] python: $PYTHON_BIN"
echo "[info] venv fallback: $VENV_DIR"
echo "[info] torch channel: $TORCH_CHANNEL"
echo "[info] requirements: $REQUIREMENTS_FILE"
echo "[info] extra requirements: $EXTRA_REQUIREMENTS"
echo "[info] tmux mode: $RUN_IN_TMUX"

if [ "$USE_CONDA" != "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

run_python -m pip install --upgrade pip setuptools wheel

case "$TORCH_CHANNEL" in
  cpu)
    run_python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu118|cu121|cu124|cu126)
    run_python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
    ;;
  *)
    echo "[error] Unsupported TORCH_CHANNEL: $TORCH_CHANNEL" >&2
    echo "[error] Expected one of: cpu, cu118, cu121, cu124, cu126" >&2
    exit 1
    ;;
esac

run_python -m pip install -r "$REQUIREMENTS_FILE" --extra-index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

for extra_file in $EXTRA_REQUIREMENTS; do
  if [ -f "$extra_file" ]; then
    run_python -m pip install -r "$extra_file" --extra-index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}
  else
    echo "[warn] extra requirements file not found, skipping: $extra_file"
  fi
done

if [ -f "$ROOT_DIR/check_sam3_import.py" ]; then
  echo "[info] running SAM3 import check"
  run_python "$ROOT_DIR/check_sam3_import.py"
fi

if [ "$USE_CONDA" = "1" ]; then
  echo "[done] Python dependencies installed in conda env $CONDA_ENV"
else
  echo "[done] Python dependencies installed in $VENV_DIR"
fi
