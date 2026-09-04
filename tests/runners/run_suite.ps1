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
        PositiveFixtures = @()
        NegativeFixtures = @(
            @{
                Path = "tests/fixtures/e2e/test_e2e_partial_install_failure.ipynb"
                ExpectedPattern = "ModuleNotFoundError: No module named 'fake_pkg_does_not_exist_xyz123'"
                ExpectedDiagnostic = "fake_pkg_does_not_exist_xyz123 failed to install"
            }
        )
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

# Where each tier's Build-DockerCmd writes the executed notebook (needed to
# recover cell-output-only diagnostics, which never reach container stdout)
$OUTPUT_NOTEBOOK = @{
    "plain" = "/tmp/out.ipynb"
    "kaggle" = "/tmp/executed_kaggle.ipynb"
    "colab" = "/tmp/out.ipynb"
}

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

# Positive Fixtures (Expected to PASS with exit code 0)
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
            -e PIP_ROOT_USER_ACTION=ignore `
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

# Negative Fixtures (Expected to FAIL with non-zero exit code and verified diagnostics)
foreach ($item in $Config.NegativeFixtures) {
    $nb = $item.Path
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect FAIL): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $baseCmd = Build-DockerCmd $Tier $nb $merged_nb
    $outputNotebook = $OUTPUT_NOTEBOOK[$Tier]

    # ExpectedDiagnostic text is printed by the generated setup cell and only
    # ever lands inside the executed notebook's cell output JSON - nbconvert
    # does not stream per-cell stdout to the container's own stdout/stderr.
    # So: run the chain, capture its real exit code, then unconditionally cat
    # the output notebook (whether or not it exists) so that text is present
    # in $rawOutput too, and finally exit with the original code so $dockerExit
    # still reflects the notebook execution's pass/fail.
    $cmd = "{ $baseCmd; } 2>&1; NB_EXIT=`$?; cat `"$outputNotebook`" 2>&1; exit `$NB_EXIT"

    try {
        $rawOutput = docker run --rm `
            --pull missing `
            -v "${REPO_ROOT}:/workspace" `
            -w /workspace `
            -e PYTHONPATH="/workspace" `
            -e PYTHONUNBUFFERED=1 `
            -e PYDEVD_DISABLE_FILE_VALIDATION=1 `
            -e PIP_ROOT_USER_ACTION=ignore `
            --entrypoint /bin/bash `
            $IMAGE `
            -c $cmd

        $dockerExit = $LASTEXITCODE
    }
    finally {
        Cleanup-Artifacts $nb
    }

    # Print captured output to console for full visibility
    if ($rawOutput) {
        $rawOutput | ForEach-Object { Write-Host $_ }
    }

    if ($dockerExit -eq 0) {
        Write-Error "Tier test $nb was expected to fail, but exited cleanly with code 0."
        exit 1
    }

    $outputStr = if ($rawOutput) { $rawOutput -join "`n" } else { "" }
    $cleanOutput = $outputStr -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''

    if ($item.ExpectedPattern -and (-not $cleanOutput.Contains($item.ExpectedPattern))) {
        Write-Error "Tier test $nb failed, but output did not contain expected pattern: $($item.ExpectedPattern)"
        exit 1
    }

    if ($item.ExpectedDiagnostic -and (-not $cleanOutput.Contains($item.ExpectedDiagnostic))) {
        Write-Error "Tier test $nb failed, but output did not contain expected diagnostic: $($item.ExpectedDiagnostic)"
        exit 1
    }

    Write-Host ">>> PASS (Controlled failure verified): $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " $($Tier.ToUpper()) TIER: ALL TESTS PASSED"
Write-Host "============================================================"