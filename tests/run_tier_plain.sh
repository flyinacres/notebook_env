#!/usr/bin/env bash
set -euo pipefail

IMAGE="python:3.11-slim"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONPATH="/workspace" \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  bash -c '
    set -euo pipefail
    
    # 1. Provision standard execution tooling globally
    pip install --no-cache-dir ipykernel nbconvert==7.17.1
    python -m ipykernel install --user --name python3

    # 2. Run targeted execution fixtures
    FIXTURES=(
      "fixtures/execution_safe/pinned_install.ipynb"
      "fixtures/execution_safe/platform_pseudo_module.ipynb"
    )

    for nb in "${FIXTURES[@]}"; do
      echo "=== [PLAIN] Processing: ${nb} ==="
      python notebook_env.py "${nb}" --output
      jupyter nbconvert --to notebook --execute "${nb%.ipynb}_merged.ipynb" --output "/tmp/out.ipynb"
      echo ">>> PASS: ${nb}"
    done
  '