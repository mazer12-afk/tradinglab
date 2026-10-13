#!/usr/bin/env bash
# Idempotent environment setup: creates .venv, installs torch (CPU wheel when the
# PyTorch index is reachable, otherwise the PyPI wheel) and the lab itself.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    || .venv/bin/pip install -q "torch>=2.2"
fi
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import torch, tradinglab; print('tradinglab', tradinglab.__version__, 'torch', torch.__version__)"
