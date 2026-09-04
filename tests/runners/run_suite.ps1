param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("python3.11", "kaggle", "colab")]
    [string]$Tier
)

$ErrorActionPreference = "Stop"

# Repository root calculation (two levels up from tests/runners/)
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path

# Tier definitions and image mappings
# PositiveFixtures/NegativeFixtures share one shape: Path (required),
# VerifyPattern (optional, checked against captured output on success),
# ExpectedPattern (optional, negative-only, checked against captured output on failure).
$TIER_CONFIG = @{
    "python3.11" = @{
        Image = "python:3.11-slim"
        PositiveFixtures = @()
        NegativeFixtures = @(
            @{
                Path = "tests/fixtures/e2e/test_e2e_partial_install_failure.ipynb"
                ExpectedPattern = "ModuleNotFoundError: No module named 'fake_pkg_does_not_exist_xyz123'"
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

function Strip-AnsiCodes($text) {
    return $text -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
}

function Build-DockerCmd($tierName, $nb, $mergedNb) {
    switch ($tierName) {
        "python3.11" {
            return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize && " + `
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

# Positive Fixtures (Expected to PASS with exit code 0, and optionally verified content)
foreach ($item in $Config.PositiveFixtures) {
    $nb = $item.Path
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect PASS): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $baseCmd = Build-DockerCmd $Tier $nb $merged_nb
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
            $IMAGE `
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
        if (-not $cleanOutput.Contains($item.VerifyPattern)) {
            Write-Error "Tier test $nb passed, but output did not contain expected pattern: $($item.VerifyPattern)"
        }
    }

    Write-Host ">>> PASS: $nb"
}

# Negative Fixtures (Expected to FAIL with non-zero exit code and verified diagnostics)
foreach ($item in $Config.NegativeFixtures) {
    $nb = $item.Path
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect FAIL): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $baseCmd = Build-DockerCmd $Tier $nb $merged_nb

    # NOTE: ExpectedDiagnostic (setup-cell print output) cannot be verified here.
    # nbconvert only writes the --output notebook file on a clean run; on
    # CellExecutionError it raises before serializing, so there is no file to
    # inspect for that text. Only ExpectedPattern (the exception traceback,
    # which nbconvert does echo to stderr) is checked below.
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
    }

    $outputStr = if ($rawOutput) { $rawOutput -join "`n" } else { "" }
    $cleanOutput = Strip-AnsiCodes $outputStr

    if ($item.ExpectedPattern -and (-not $cleanOutput.Contains($item.ExpectedPattern))) {
        Write-Error "Tier test $nb failed, but output did not contain expected pattern: $($item.ExpectedPattern)"
    }

    Write-Host ">>> PASS (Controlled failure verified): $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " $($Tier.ToUpper()) TIER: ALL TESTS PASSED"
Write-Host "============================================================"