    Write-Host "Running Phase 5k: Local Package Pin-and-Verify (1.0.0 -> 2.0.0)..." -ForegroundColor Cyan
    $repinNb = "tests/fixtures/e2e/test_local_pkg_pin_and_verify.ipynb"
    $repinMerged = $repinNb -replace '\.ipynb$', '_merged.ipynb'
    $repinCmd = "pip install --no-cache-dir ipykernel nbconvert==7.17.1 -q && " + `
                "python -m ipykernel install --user --name python3 && " + `
                "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist pip install local_test_pkg==1.0.0 && " + `
                "python notebook_env.py `"$repinNb`" --output && " + `
                "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist jupyter nbconvert --to notebook --execute `"$repinMerged`" --output /tmp/out_v1.ipynb --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3 && " + `
                "cat /tmp/out_v1.ipynb && " + `
                "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist pip install local_test_pkg==2.0.0 && " + `
                "python notebook_env.py `"$repinNb`" --output && " + `
                "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist jupyter nbconvert --to notebook --execute `"$repinMerged`" --output /tmp/out_v2.ipynb --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3 && " + `
                "cat /tmp/out_v2.ipynb"

    try {
        $repinOutput = docker run --rm --pull missing `
            -v "${REPO_ROOT}:/workspace" `
            -w /workspace `
            -e PYTHONPATH="/workspace" `
            -e PYTHONUNBUFFERED=1 `
            -e PIP_ROOT_USER_ACTION=ignore `
            --entrypoint /bin/bash `
            python:3.11-slim `
            -c "$repinCmd 2>&1"

        $repinExit = $LASTEXITCODE
    }
    finally {
        Cleanup-Artifacts $repinNb
    }

    if ($repinOutput) {
        $repinOutput | ForEach-Object { Write-Host $_ }
    }

    if ($repinExit -ne 0) {
        Write-Error "Local package pin-and-verify test failed to execute cleanly"
    }

    $repinClean = Strip-AnsiCodes ($repinOutput -join "`n")
    foreach ($pattern in @(
        "Pinned install verification passed: local_test_pkg 1.0.0 active",
        "Pinned install verification passed: local_test_pkg 2.0.0 active"
    )) {
        if (-not $repinClean.Contains($pattern)) {
            Write-Error "Local package pin-and-verify test did not contain expected pattern: $pattern"
        }
    }
    Write-Host "PASS: Local package pin-and-verify test`n" -ForegroundColor Green
