param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("plain", "kaggle", "colab")]
    [string]$Tier
)

$ErrorActionPreference = "Stop"

# Repository root calculation (two levels up from tests/runners/)
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path

# Tier definitions and image mappings
$TIER_CONFIG = @{
    "plain" = @{
        Image = "python:3.11-slim"
        PositiveFixtures = @(
            "tests/fixtures/e2e/test_pip_satisfied.ipynb",
            "tests/fixtures/e2e/pinned_install.ipynb",
            "tests/fixtures/e2e/platform_pseudo_module.ipynb"
        )
        NegativeFixtures = @()
    }
    "kaggle" = @{
        Image = "gcr.io/kaggle-images/python:latest"
        PositiveFixtures = @(
            "tests/fixtures/e2e/test_pip_satisfied.ipynb",
            "tests/fixtures/e2e/pinned_install.ipynb",
            "tests/fixtures/e2e/platform_pseudo_module.ipynb"
        )
        NegativeFixtures = @()
    }
    "colab" = @{
        Image = "us-docker.pkg.dev/colab-images/public/cpu-runtime:latest"
        PositiveFixtures = @(
            "tests/fixtures/e2e/test_pip_satisfied.ipynb",
            "tests/fixtures/e2e/pinned_install.ipynb",
            "tests/fixtures/e2e/platform_pseudo_module.ipynb"
        )
        NegativeFixtures = @()
    }
}

$Config = $TIER_CONFIG[$Tier]
$IMAGE = $Config.Image

Write-Host "============================================================"
Write-Host " Running E2E Suite: $($Tier.ToUpper()) TIER ($IMAGE)"
Write-Host "============================================================"

function Cleanup-Artifacts($nbPath) {
    $merged = $nbPath -replace '\.ipynb$', '_merged.ipynb'
    $localMerged = Join-Path $REPO_ROOT $merged
    if (Test-Path $localMerged) {
        Remove-Item -Force $localMerged
    }
}

function Build-DockerCmd($tierName, $nb, $mergedNb) {
    switch ($tierName) {
        "plain" {
            return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize && " + `
                   "python -m ipykernel install --user --name python3 && " + `
                   "python notebook_env.py `"$nb`" --output && " + `
                   "jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/out.ipynb`""
        }
        "kaggle" {
            $cmd = "python3 -m venv --system-site-packages --without-pip --clear /tmp/run_env && " + `
                   "/tmp/run_env/bin/python notebook_env.py `"$nb`" --output && " + `
                   "/tmp/run_env/bin/python -m jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/executed_kaggle.ipynb`""
            if ($nb -match "tests_real_install") {
                $cmd += " && grep -q 'Real install succeeded' /tmp/executed_kaggle.ipynb"
            }
            return $cmd
        }
        "colab" {
            return "python3 notebook_env.py `"$nb`" --output && " + `
                   "jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/out.ipynb`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        }
    }
}

# Positive Fixtures (Expected to PASS)
foreach ($nb in $Config.PositiveFixtures) {
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect PASS): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $cmd = Build-DockerCmd $Tier $nb $merged_nb

    try {
        docker run --rm `
            --pull missing `
            -v "${REPO_ROOT}:/workspace" `
            -w /workspace `
            -e PYTHONPATH="/workspace" `
            -e PYTHONUNBUFFERED=1 `
            -e PYDEVD_DISABLE_FILE_VALIDATION=1 `
            --entrypoint /bin/bash `
            $IMAGE `
            -c $cmd

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Tier execution failed unexpectedly on positive test $nb"
            exit $LASTEXITCODE
        }
        Write-Host ">>> PASS: $nb"
    }
    finally {
        Cleanup-Artifacts $nb
    }
}

# Negative Fixtures (Expected to FAIL with non-zero exit code)
foreach ($nb in $Config.NegativeFixtures) {
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect FAIL): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $cmd = Build-DockerCmd $Tier $nb $merged_nb

    try {
        docker run --rm `
            --pull missing `
            -v "${REPO_ROOT}:/workspace" `
            -w /workspace `
            -e PYTHONPATH="/workspace" `
            -e PYTHONUNBUFFERED=1 `
            -e PYDEVD_DISABLE_FILE_VALIDATION=1 `
            --entrypoint /bin/bash `
            $IMAGE `
            -c $cmd

        if ($LASTEXITCODE -eq 0) {
            Write-Error "Tier test $nb was expected to fail, but exited cleanly with code 0."
            exit 1
        }
        Write-Host ">>> PASS (Controlled failure verified): $nb"
    }
    finally {
        Cleanup-Artifacts $nb
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " $($Tier.ToUpper()) TIER: ALL TESTS PASSED"
Write-Host "============================================================"