#!/usr/bin/env bash
set -euo pipefail

IMAGE="us-docker.pkg.dev/colab-images/public/runtime:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm \
  --platform linux/amd64 \
  --entrypoint /bin/bash \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONPATH="/workspace" \
  -e PYDEVD_DISABLE_FILE_VALIDATION=1 \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  -c '
    set -euo pipefail

    FIXTURES=(
      "fixtures/execution_safe/clean_baseline.ipynb"
      "fixtures/execution_safe/pinned_install.ipynb"
      "fixtures/execution_safe/platform_pseudo_module.ipynb"
    )

    for nb in "${FIXTURES[@]}"; do
      echo "=== [COLAB] Processing: ${nb} ==="
      python3 notebook_env.py "${nb}" --output
      jupyter nbconvert --to notebook --execute "${nb%.ipynb}_merged.ipynb" --output "/tmp/out.ipynb"
      echo ">>> PASS: ${nb}"
    done
  '