$ErrorActionPreference = "Stop"

$IMAGE = "us-docker.pkg.dev/colab-images/public/cpu-runtime:latest"
$WORKSPACE = $PWD.Path

$FIXTURES = @(
    "fixtures/execution_safe/test_pip_satisfied.ipynb",
    "fixtures/execution_safe/test_pip_unmanaged_reinstall.ipynb"
)

Write-Host "=== Pulling Colab CPU Runtime Image ==="
docker pull $IMAGE

foreach ($nb in $FIXTURES) {
    Write-Host "=== [COLAB] Processing: $nb ==="
    
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'

    docker run --rm `
        -v "${WORKSPACE}:/workspace" `
        -w /workspace `
        -e PYDEVD_DISABLE_FILE_VALIDATION=1 `
        -e PYTHONUNBUFFERED=1 `
        --entrypoint /bin/bash `
        $IMAGE `
        -c "python3 notebook_env.py `"$nb`" --output && jupyter nbconvert --to notebook --execute `"$merged_nb`" --output `"/tmp/out.ipynb`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Execution failed for $nb"
        exit $LASTEXITCODE
    }

    Write-Host ">>> PASS: $nb"
}

Write-Host "All Colab tier execution tests completed successfully."