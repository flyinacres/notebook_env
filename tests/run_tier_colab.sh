#!/usr/bin/env bash
set -eo pipefail

IMAGE="us-docker.pkg.dev/colab-images/public/runtime:latest"
WORKSPACE="$(pwd)"

FIXTURES=(
  "fixtures/execution_safe/clean_baseline.ipynb"
  "fixtures/execution_safe/pinned_install.ipynb"
  "fixtures/execution_safe/platform_pseudo_module.ipynb"
)

echo "=== Pulling Colab Runtime Image ==="
docker pull --platform linux/amd64 "${IMAGE}"

for nb in "${FIXTURES[@]}"; do
  echo "=== [COLAB] Processing: ${nb} ==="
  
  merged_nb="${nb%.ipynb}_merged.ipynb"

  # Run both analysis/generation AND execution inside the container
  docker run --rm \
    --platform linux/amd64 \
    -v "${WORKSPACE}:/workspace" \
    -w /workspace \
    -e PYDEVD_DISABLE_FILE_VALIDATION=1 \
    -e PYTHONUNBUFFERED=1 \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -c "python3 notebook_env.py \"${nb}\" --output && \
        jupyter nbconvert \
          --to notebook \
          --execute \"${merged_nb}\" \
          --output \"/tmp/out.ipynb\" \
          --ExecutePreprocessor.timeout=300 \
          --ExecutePreprocessor.kernel_name=python3"
      
  echo ">>> PASS: ${nb}"
done

echo "All Colab tier execution tests completed successfully."