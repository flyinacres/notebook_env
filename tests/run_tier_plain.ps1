$ErrorActionPreference = "Stop"

$IMAGE = "python:3.11-slim"
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "============================================================"
Write-Host " Running Axis B Suite: PLAIN TIER ($IMAGE)"
Write-Host "============================================================"

$FIXTURES = @(
    "fixtures/execution_safe/pinned_install.ipynb",
    "fixtures/execution_safe/platform_pseudo_module.ipynb"
)

foreach ($nb in $FIXTURES) {
    Write-Host ""
    Write-Host "=== [PLAIN] Processing: $nb ==="

    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'

    $cmd = "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize && " + `
           "python -m ipykernel install --user --name python3 && " + `
           "python notebook_env.py `"$nb`" --output && " + `
           "jupyter nbconvert --to notebook --execute `"$merged_nb`" --output `"/tmp/out.ipynb`""

    docker run --rm `
        -v "${REPO_ROOT}:/workspace" `
        -w /workspace `
        -e PYTHONPATH="/workspace" `
        -e PYTHONUNBUFFERED=1 `
        --entrypoint /bin/bash `
        $IMAGE `
        -c $cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Plain tier execution failed on $nb"
        exit $LASTEXITCODE
    }

    Write-Host ">>> PASS: $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " PLAIN TIER: ALL TESTS PASSED"
Write-Host "============================================================"