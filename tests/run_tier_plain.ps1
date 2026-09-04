$ErrorActionPreference = "Stop"

$IMAGE = "python:3.11-slim"
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "============================================================"
Write-Host " Running Axis B Suite: PLAIN TIER ($IMAGE)"
Write-Host "============================================================"

$POSITIVE_FIXTURES = @(
    "fixtures/execution_safe/pinned_install.ipynb",
    "fixtures/execution_safe/platform_pseudo_module.ipynb"
)

$NEGATIVE_FIXTURES = @(
    "fixtures/test_negative_bad_package.ipynb"
)

foreach ($nb in $POSITIVE_FIXTURES) {
    Write-Host ""
    Write-Host "=== [PLAIN] Processing (Expect PASS): $nb ==="

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
        Write-Error "Plain tier execution failed unexpectedly on positive test $nb"
        exit $LASTEXITCODE
    }

    Write-Host ">>> PASS: $nb"
}

foreach ($nb in $NEGATIVE_FIXTURES) {
    Write-Host ""
    Write-Host "=== [PLAIN] Processing (Expect FAIL): $nb ==="

    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'

    $cmd = "pip install --no-cache-dir ipykernel nbconvert==7.17.1 && " + `
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

    if ($LASTEXITCODE -eq 0) {
        Write-Error "Plain tier test $nb was expected to fail, but passed cleanly."
        exit 1
    }

    Write-Host ">>> PASS (Controlled failure verified): $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " PLAIN TIER: ALL TESTS PASSED"
Write-Host "============================================================"