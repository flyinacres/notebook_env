$ErrorActionPreference = "Stop"

$IMAGE = "gcr.io/kaggle-images/python:latest"
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "============================================================"
Write-Host " Running Axis B Suite: KAGGLE TIER ($IMAGE)"
Write-Host "============================================================"

$FIXTURES = @(
    "fixtures/test_scenario3_multicell.ipynb"
)

foreach ($nb in $FIXTURES) {
    Write-Host ""
    Write-Host "--- [KAGGLE] Processing: $nb ---"

    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'

    $cmd = "python3 -m venv --system-site-packages --without-pip --clear /tmp/run_env && " + `
           "/tmp/run_env/bin/python notebook_env.py `"$nb`" --output && " + `
           "echo '--- [KAGGLE] Executing merged notebook: $merged_nb ---' && " + `
           "/tmp/run_env/bin/python -m jupyter nbconvert --to notebook --execute `"$merged_nb`" --output `"/tmp/executed_kaggle.ipynb`""

    if ($nb -match "tests_real_install") {
        $cmd += " && grep -q 'Real install succeeded' /tmp/executed_kaggle.ipynb"
    }

    docker run --rm `
        -v "${REPO_ROOT}:/workspace" `
        -w /workspace `
        -e PYTHONPATH="/workspace" `
        --entrypoint /bin/bash `
        $IMAGE `
        -c $cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Kaggle tier execution failed on $nb"
        exit $LASTEXITCODE
    }

    Write-Host ">>> PASS: $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " KAGGLE TIER: ALL TESTS PASSED"
Write-Host "============================================================"