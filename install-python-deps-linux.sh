#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-linux}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu126}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements-linux-lock.txt"
RUN_IN_TMUX="${RUN_IN_TMUX:-0}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-bme-python-deps}"

start_in_tmux() {
  local command
  command=$(printf 'cd %q && RUN_IN_TMUX=0 PYTHON_BIN=%q VENV_DIR=%q TORCH_CHANNEL=%q bash %q' \
    "$ROOT_DIR" "$PYTHON_BIN" "$VENV_DIR" "$TORCH_CHANNEL" "$ROOT_DIR/install-python-deps-linux.sh")
  tmux new-session -d -s "$TMUX_SESSION_NAME" "$command"
  echo "[info] started tmux session: $TMUX_SESSION_NAME"
  echo "[info] attach with: tmux attach -t $TMUX_SESSION_NAME"
  exit 0
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[info] root: $ROOT_DIR"
echo "[info] python: $PYTHON_BIN"
echo "[info] venv: $VENV_DIR"
echo "[info] torch channel: $TORCH_CHANNEL"
echo "[info] tmux mode: $RUN_IN_TMUX"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "$TORCH_CHANNEL" in
  cpu)
    python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu118|cu121|cu124|cu126)
    python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
    ;;
  *)
    echo "[error] Unsupported TORCH_CHANNEL: $TORCH_CHANNEL" >&2
    echo "[error] Expected one of: cpu, cu118, cu121, cu124, cu126" >&2
    exit 1
    ;;
esac

python -m pip install -r "$REQUIREMENTS_FILE" --extra-index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

if [ -f "$ROOT_DIR/check_sam3_import.py" ]; then
  echo "[info] running SAM3 import check"
  python "$ROOT_DIR/check_sam3_import.py"
fi

echo "[done] Python dependencies installed in $VENV_DIR"