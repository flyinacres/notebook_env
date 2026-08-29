#!/usr/bin/env bash
set -euo pipefail

IMAGE="us-docker.pkg.dev/colab-images/public/runtime:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_NB="fixtures/execution_safe/clean_baseline.ipynb"

echo "============================================================"
echo " Verifying Colab Headless Execution (${IMAGE})"
echo " Target Notebook: ${TARGET_NB}"
echo "============================================================"

docker run --rm \
  --platform linux/amd64 \
  --entrypoint /bin/bash \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONPATH="/workspace" \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  -c "
    set -euo pipefail

    echo '=== Step 1: Run notebook_env.py analysis and generation ==='
    python3 notebook_env.py \"${TARGET_NB}\" --output

    MERGED_NB=\"${TARGET_NB%.ipynb}_merged.ipynb\"
    OUTPUT_NB=\"/tmp/colab_verified.ipynb\"

    echo '=== Step 2: Execute generated companion notebook headlessly ==='
    python3 -m jupyter nbconvert \
      --to notebook \
      --execute \"\${MERGED_NB}\" \
      --output \"\${OUTPUT_NB}\" \
      --ExecutePreprocessor.timeout=60 \
      --ExecutePreprocessor.extra_arguments=\"['--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel', '--InteractiveShellApp.extensions=[]']\"

    echo '=== Step 3: Confirm execution completed ==='
    test -f \"\${OUTPUT_NB}\"
    echo '>>> SUCCESS: Colab container executed cleanly without hanging.'
  "