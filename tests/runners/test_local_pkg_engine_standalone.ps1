[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path
$image = "python:3.11-slim"
$outputPath = "/tmp/out_local_pkg.ipynb"

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

function Build-LocalPkgDockerCmd($nb, $mergedNb) {
    return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 -q && " + `
           "python -m ipykernel install --user --name python3 && " + `
           "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/bootstrap pip install --no-cache-dir setuptools wheel && " + `
           "SEED_VER=`$(grep -oE 'local_test_pkg==[0-9.]+' `"$nb`" | head -1 | cut -d= -f3) && " + `
           "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/seed_dist pip install --no-cache-dir local_test_pkg==`$SEED_VER && " + `
           "python notebook_env.py `"$nb`" --output --timeout 2 && " + `
           "pip uninstall -y local_test_pkg && " + `
           "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist PIP_NO_BUILD_ISOLATION=1 PIP_NO_CACHE_DIR=1 jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"$outputPath`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
}

$fixtures = @(
    @{
        Path = "tests/fixtures/e2e/test_local_pkg_timeout.ipynb"
        VerifyPattern = @("Installation timed out after 2s.")
    },
    @{
        Path = "tests/fixtures/e2e/test_local_pkg_build_failure.ipynb"
        VerifyPattern = @("RuntimeError: deliberate build failure for local_test_pkg")
    }
)

foreach ($item in $fixtures) {
    $nb = $item.Path
    Write-Host ""
    Write-Host "=== Processing (Expect PASS): $nb ==="
    $merged_nb = $nb -replace '\.ipynb$', '_merged.ipynb'
    $baseCmd = Build-LocalPkgDockerCmd $nb $merged_nb
    $baseCmd = "$baseCmd && cat `"$outputPath`""
    $cmd = "$baseCmd 2>&1"

    try {
        $rawOutput = docker run --rm `
            --pull missing `
            -v "${REPO_ROOT}:/workspace" `
            -w /workspace `
            -e PYTHONPATH="/workspace" `
            -e PYTHONUNBUFFERED=1 `
            -e PIP_ROOT_USER_ACTION=ignore `
            --entrypoint /bin/bash `
            $image `
            -c $cmd

        $dockerExit = $LASTEXITCODE
    }
    finally {
        Cleanup-Artifacts $nb
    }

    if ($rawOutput) {
        $rawOutput | ForEach-Object { Write-Host $_ }
    }

    if ($dockerExit -ne 0) {
        Write-Error "Tier execution failed unexpectedly on positive test $nb"
    }

    $outputStr = if ($rawOutput) { $rawOutput -join "`n" } else { "" }
    $cleanOutput = Strip-AnsiCodes $outputStr
    foreach ($pattern in @($item.VerifyPattern)) {
        if (-not $cleanOutput.Contains($pattern)) {
            Write-Error "Tier test $nb passed, but output did not contain expected pattern: $pattern"
        }
    }

    Write-Host ">>> PASS: $nb"
}
