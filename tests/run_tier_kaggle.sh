#!/usr/bin/env bash
set -euo pipefail

IMAGE="gcr.io/kaggle-images/python:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo " Running Axis B Suite: KAGGLE TIER (${IMAGE})"
echo "============================================================"

docker run --rm \
  --platform linux/amd64 \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONPATH="/workspace" \
  "${IMAGE}" \
  bash -c '
    set -euo pipefail

    FIXTURES=(
      "fixtures/execution_safe/clean_baseline.ipynb"
      "fixtures/execution_safe/pinned_install.ipynb"
      "fixtures/execution_safe/platform_pseudo_module.ipynb"
    )

   for nb in "${FIXTURES[@]}"; do
      echo ""
      echo "--- [KAGGLE] Processing: ${nb} ---"

      # 1. Fresh venv retaining system site packages without crashing on ensurepip
      python3 -m venv --system-site-packages --without-pip --clear /tmp/run_env

      # 2. Run notebook_env.py in --output mode
      /tmp/run_env/bin/python notebook_env.py "${nb}" --output

      merged_nb="${nb%.ipynb}_merged.ipynb"

      # 3. Execute generated notebook using the environment python
      echo "--- [KAGGLE] Executing merged notebook: ${merged_nb} ---"
      /tmp/run_env/bin/python -m jupyter nbconvert \
        --to notebook \
        --execute "${merged_nb}" \
        --output "/tmp/executed_kaggle.ipynb"

      echo ">>> PASS: ${nb}"
    done

    echo ""
    echo "============================================================"
    echo " KAGGLE TIER: ALL TESTS PASSED"
    echo "============================================================"
  '
