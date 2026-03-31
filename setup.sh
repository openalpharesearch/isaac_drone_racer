#!/usr/bin/env bash
# Copyright (c) 2025, Ai Robotics @ Berkeley
# SPDX-License-Identifier: BSD-3-Clause
#
# (Re)create a self-contained venv with pinned Isaac Sim 4.5 (pip pkg
# 4.5.0), IsaacLab, PyTorch and all project deps.  Safe to re-run — the
# old venv is removed first.
#
# Usage:
#   ./setup.sh                    # fresh install or rebuild
#   source env_isaaclab/bin/activate  # then activate the venv
set -euo pipefail

VENV_DIR="env_isaaclab"

# ── prerequisites ───────────────────────────────────────────────────
for cmd in uv python3.10; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found. Install it first."; exit 1; }
done

# ── clean slate ─────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "Removing existing $VENV_DIR …"
  rm -rf "$VENV_DIR"
fi

# ── create venv + install ──────────────────────────────────────────
uv venv --python 3.10 --seed "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install -e .

echo ""
echo "Done.  Activate with:  source $VENV_DIR/bin/activate"
