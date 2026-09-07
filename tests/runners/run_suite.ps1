[CmdletBinding(DefaultParameterSetName = "SingleTier")]
param (
    [Parameter(Mandatory = $true, ParameterSetName = "SingleTier")]
    [ValidateSet("common", "python3.11", "kaggle", "colab")]
    [string]$Tier,

    [Parameter(Mandatory = $true, ParameterSetName = "AllTiers")]
    [switch]$All
)

$ErrorActionPreference = "Stop"

# Repository root calculation (two levels up from tests/runners/)
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path

# Tier definitions and image mappings
# PositiveFixtures/NegativeFixtures share one shape: Path (required),
# VerifyPattern (optional, positive-only, checked against the executed notebook's content),
# ExpectedEname/ExpectedEvalueSubstring (optional, negative-only, checked against
# the single structural error parsed out of the executed notebook - see the
# negative-fixture loop below for why this replaced a plain traceback grep).
#
# "common" (tier-independent tests, e.g. the live-kernel script) is deliberately
# NOT an entry here - it's a structurally different kind of test, not just a
# different Docker image, so it gets its own function (Invoke-CommonTests)
# rather than being forced into this fixture-list shape.
$TIER_CONFIG = @{
    "python3.11" = @{
        Image = "python:3.11-slim"
        PositiveFixtures = @(
            @{
                Path = "tests/fixtures/e2e/test_partial_install_recovery.ipynb"
                VerifyPattern = @(
                    "Partial install succeeded for valid packages: humanize==4.16.0, tabulate==0.9.0",
                    "tabulate==0.0.0.nonexistent failed to install"
                )
            },
            @{
                Path = "tests/fixtures/e2e/test_numpy_old_pin_preserves_api.ipynb"
                VerifyPattern = @(
                    "numpy==1.23.5: numpy.bool alias still works as expected"
                )
            }
        )
        NegativeFixtures = @(
            @{
                Path = "tests/fixtures/e2e/test_e2e_failed_repin_surfaces_downstream.ipynb"
                ExpectedEname = "AssertionError"
                ExpectedEvalueSubstring = "failed silently"
            }
        )
    }
    "kaggle" = @{
        Image = "gcr.io/kaggle-images/python:latest"
        PositiveFixtures = @(
            @{ Path = "tests/fixtures/e2e/test_pip_satisfied.ipynb" },
            @{ Path = "tests/fixtures/e2e/pinned_install.ipynb" },
            @{ Path = "tests/fixtures/e2e/platform_pseudo_module.ipynb" }
        )
        NegativeFixtures = @()
    }
    "colab" = @{
        Image = "us-docker.pkg.dev/colab-images/public/cpu-runtime:latest"
        PositiveFixtures = @(
            @{ Path = "tests/fixtures/e2e/test_pip_satisfied.ipynb" },
            @{ Path = "tests/fixtures/e2e/pinned_install.ipynb" },
            @{ Path = "tests/fixtures/e2e/platform_pseudo_module.ipynb" }
        )
        NegativeFixtures = @()
    }
}

$OUTPUT_NOTEBOOK_PATH = @{
    "python3.11" = "/tmp/out.ipynb"
    "kaggle" = "/tmp/executed_kaggle.ipynb"
    "colab" = "/tmp/out.ipynb"
}

function Cleanup-Artifacts($nbPath) {
    $merged = $nbPath -replace '\.ipynb$', '_merged.ipynb'
    $localMerged = Join-Path $REPO_ROOT $merged
    if (Test-Path $localMerged) {
        Remove-Item -Force $localMerged
    }
}

function Strip-AnsiCodes($text) {
    return $text -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
}

function Build-DockerCmd($tierName, $nb, $mergedNb) {
    switch ($tierName) {
        "python3.11" {
            return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize==4.16.0 tabulate==0.9.0 numpy==1.23.5 && " + `
                   "python -m ipykernel install --user --name python3 && " + `
                   "python notebook_env.py `"$nb`" --output && " + `
                   "jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/out.ipynb`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        }
        "kaggle" {
            return "python3 -m venv --system-site-packages --without-pip --clear /tmp/run_env && " + `
                   "/tmp/run_env/bin/python notebook_env.py `"$nb`" --output && " + `
                   "/tmp/run_env/bin/python -m jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/executed_kaggle.ipynb`" --ExecutePreprocessor.timeout=300"
        }
        "colab" {
            return "python3 notebook_env.py `"$nb`" --output && " + `
                   "jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/out.ipynb`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        }
        default {
            throw "Build-DockerCmd: no command defined for tier '$tierName'"
        }
    }
}

function Invoke-CommonTests {
    Write-Host "============================================================"
    Write-Host " Running tier-independent tests"
    Write-Host "============================================================"

    Write-Host "Running Phase 5g: Live-Kernel Stale Module Test..." -ForegroundColor Cyan
    docker run --rm --pull missing -v "${REPO_ROOT}:/workspace" -w /workspace -e PYTHONPATH="/workspace" -e PYTHONUNBUFFERED=1 -e PIP_ROOT_USER_ACTION=ignore --entrypoint /bin/bash python:3.11-slim -c "pip install --no-cache-dir jupyter_client ipykernel numpy==1.26.4 -q && python -m ipykernel install --user --name python3 && python tests/runners/test_live_kernel_stale_repin.py"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Live-kernel stale module test failed"
    }
    Write-Host "PASS: Live-kernel stale module test`n" -ForegroundColor Green

    Write-Host "Running Phase 5g: Live-Kernel Phase 0 Regressions..." -ForegroundColor Cyan
    docker run --rm --pull missing -v "${REPO_ROOT}:/workspace" -w /workspace -e PYTHONPATH="/workspace" -e PYTHONUNBUFFERED=1 -e PIP_ROOT_USER_ACTION=ignore --entrypoint /bin/bash python:3.11-slim -c "pip install --no-cache-dir jupyter_client ipykernel -q && python -m ipykernel install --user --name python3 && python tests/runners/test_live_kernel_phase0_regressions.py"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Live-kernel Phase 0 regressions test failed"
    }
    Write-Host "PASS: Live-kernel Phase 0 regressions test`n" -ForegroundColor Green
}

function Invoke-TierTests($tierName) {
    $config = $TIER_CONFIG[$tierName]
    $image = $config.Image

    Write-Host "============================================================"
    Write-Host " Running E2E Suite: $($tierName.ToUpper()) TIER ($image)"
    Write-Host "============================================================"

    # Positive Fixtures (Expected to PASS with exit code 0, and optionally verified content)
    foreach ($item in $config.PositiveFixtures) {
        $nb = $item.Path
        Write-Host ""
        Write-Host "=== [$($tierName.ToUpper())] Processing (Expect PASS): $nb ==="
        $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
        $baseCmd = Build-DockerCmd $tierName $nb $merged_nb
        if ($item.VerifyPattern) {
            # Chained with && so this only runs (and only needs to succeed) on
            # the clean-exit path already required for a positive fixture.
            $baseCmd = "$baseCmd && cat `"$($OUTPUT_NOTEBOOK_PATH[$tierName])`""
        }
        $cmd = "$baseCmd 2>&1"

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
                $image `
                -c $cmd

            $dockerExit = $LASTEXITCODE
        }
        finally {
            Cleanup-Artifacts $nb
        }

        # Print captured output to console for full visibility, pass or fail
        if ($rawOutput) {
            $rawOutput | ForEach-Object { Write-Host $_ }
        }

        if ($dockerExit -ne 0) {
            Write-Error "Tier execution failed unexpectedly on positive test $nb"
        }

        if ($item.VerifyPattern) {
            $outputStr = if ($rawOutput) { $rawOutput -join "`n" } else { "" }
            $cleanOutput = Strip-AnsiCodes $outputStr
            foreach ($pattern in @($item.VerifyPattern)) {
                if (-not $cleanOutput.Contains($pattern)) {
                    Write-Error "Tier test $nb passed, but output did not contain expected pattern: $pattern"
                }
            }
        }

        Write-Host ">>> PASS: $nb"
    }

    # Negative Fixtures (Expected to raise exactly one cell error, of a specific
    # type and message, and no other errors anywhere in the notebook)
    foreach ($item in $config.NegativeFixtures) {
        $nb = $item.Path
        Write-Host ""
        Write-Host "=== [$($tierName.ToUpper())] Processing (Expect FAIL): $nb ==="
        $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
        $baseCmd = Build-DockerCmd $tierName $nb $merged_nb
        $outputPath = $OUTPUT_NOTEBOOK_PATH[$tierName]

        # --allow-errors makes nbconvert always write the output notebook and
        # always exit 0 on a cell error, so we can inspect the actual notebook
        # JSON via check_negative_fixture.py instead of grepping a single
        # traceback out of stderr. That script does its own exactly-one-error
        # plus ename/evalue comparison and exits 0/1 accordingly, so $dockerExit
        # alone is the pass/fail signal here.
        $baseCmd = "$baseCmd --allow-errors && python3 tests/runners/check_negative_fixture.py `"$outputPath`" `"$($item.ExpectedEname)`" `"$($item.ExpectedEvalueSubstring)`""
        $cmd = "$baseCmd 2>&1"

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
                $image `
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

        if ($dockerExit -ne 0) {
            Write-Error "Tier test ${nb}: structural check failed (see PASS/FAIL line above, or an unrelated invocation failure)."
        }

        Write-Host ">>> PASS (Structural failure verified): $nb"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " $($tierName.ToUpper()) TIER: ALL TESTS PASSED"
    Write-Host "============================================================"
}

# Dispatch
if ($All) {
    Invoke-CommonTests
    foreach ($tierName in $TIER_CONFIG.Keys) {
        Invoke-TierTests $tierName
    }
}
elseif ($Tier -eq "common") {
    Invoke-CommonTests
}
else {
    Invoke-TierTests $Tier
}

Write-Host ""
Write-Host "============================================================"
Write-Host " ALL REQUESTED TESTS PASSED"
Write-Host "============================================================"