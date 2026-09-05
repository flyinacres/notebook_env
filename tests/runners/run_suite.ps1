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
# VerifyPattern (optional, positive-only, checked against the executed notebook's content),
# ExpectedEname/ExpectedEvalueSubstring (optional, negative-only, checked against
# the single structural error parsed out of the executed notebook - see the
# negative-fixture loop below for why this replaced a plain traceback grep).
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

$Config = $TIER_CONFIG[$Tier]
$IMAGE = $Config.Image

# Where each tier's Build-DockerCmd writes the executed notebook. Only used
# when a fixture declares VerifyPattern: per-cell print output lands only in
# this file's JSON, never in the container's own stdout/stderr stream.
$OUTPUT_NOTEBOOK_PATH = @{
    "python3.11" = "/tmp/out.ipynb"
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

function Strip-AnsiCodes($text) {
    return $text -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
}

function Build-DockerCmd($tierName, $nb, $mergedNb) {
    switch ($tierName) {
        "python3.11" {
            return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize==4.16.0 tabulate==0.9.0 && " + `
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
    if ($item.VerifyPattern) {
        # Chained with && so this only runs (and only needs to succeed) on
        # the clean-exit path already required for a positive fixture.
        $baseCmd = "$baseCmd && cat `"$($OUTPUT_NOTEBOOK_PATH[$Tier])`""
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
foreach ($item in $Config.NegativeFixtures) {
    $nb = $item.Path
    Write-Host ""
    Write-Host "=== [$($Tier.ToUpper())] Processing (Expect FAIL): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $baseCmd = Build-DockerCmd $Tier $nb $merged_nb
    $outputPath = $OUTPUT_NOTEBOOK_PATH[$Tier]

    # --allow-errors makes nbconvert always write the output notebook and
    # always exit 0 on a cell error, so we can inspect the actual notebook
    # JSON instead of grepping a single traceback out of stderr. The inline
    # python script below is the structural check: exactly one error output
    # across the whole notebook, reported as ERROR_ENAME=/ERROR_EVALUE=. It
    # exits 1 (breaking the chain, so $dockerExit reflects a real failure)
    # if there are zero or more than one error outputs.
    $pyCode = "import json,sys; nb=json.load(open('$outputPath')); errs=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb['cells']) for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('ERROR_COUNT='+str(len(errs))); [print('ERROR_ENAME='+str(e[1])) for e in errs[:1]]; [print('ERROR_EVALUE='+str(e[2])) for e in errs[:1]]; sys.exit(0 if len(errs)==1 else 1)"
    $pyCodeB64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($pyCode))
    $baseCmd = "$baseCmd --allow-errors && echo $pyCodeB64 | base64 -d | python3"
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

    $outputStr = if ($rawOutput) { $rawOutput -join "`n" } else { "" }
    $cleanOutput = Strip-AnsiCodes $outputStr

    if ($dockerExit -ne 0) {
        Write-Error "Tier test ${nb}: expected exactly one cell error, but the structural check failed (see ERROR_COUNT above, or an unrelated invocation failure)."
    }

    $enameMatch = [regex]::Match($cleanOutput, "ERROR_ENAME=(.*)")
    $evalueMatch = [regex]::Match($cleanOutput, "ERROR_EVALUE=(.*)")

    if ($item.ExpectedEname -and (-not $enameMatch.Success -or $enameMatch.Groups[1].Value.Trim() -ne $item.ExpectedEname)) {
        Write-Error "Tier test ${nb}: expected ename '$($item.ExpectedEname)' but got '$($enameMatch.Groups[1].Value)'"
    }

    if ($item.ExpectedEvalueSubstring -and (-not $evalueMatch.Success -or -not $evalueMatch.Groups[1].Value.Contains($item.ExpectedEvalueSubstring))) {
        Write-Error "Tier test ${nb}: expected evalue to contain '$($item.ExpectedEvalueSubstring)' but got '$($evalueMatch.Groups[1].Value)'"
    }

    Write-Host ">>> PASS (Structural failure verified): $nb"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " $($Tier.ToUpper()) TIER: ALL TESTS PASSED"
Write-Host "============================================================"