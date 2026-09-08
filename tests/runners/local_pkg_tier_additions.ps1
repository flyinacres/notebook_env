# ============================================================
# 4 insertions into run_suite.ps1's existing structure.
# ============================================================

# 1. Line 4 -- add "local_pkg" to the ValidateSet:
[ValidateSet("common", "python3.11", "kaggle", "colab", "local_pkg")]

# 2. Inside $TIER_CONFIG (after the "colab" entry, before the closing brace
#    at line 71) -- new tier definition:
    "local_pkg" = @{
        Image = "python:3.11-slim"
        PositiveFixtures = @(
            @{
                Path = "tests/fixtures/e2e/test_local_pkg_timeout.ipynb"
                VerifyPattern = @(
                    "Installation timed out after 2s."
                )
            },
            @{
                Path = "tests/fixtures/e2e/test_local_pkg_build_failure.ipynb"
                VerifyPattern = @(
                    "RuntimeError: deliberate build failure for local_test_pkg"
                )
            }
        )
        NegativeFixtures = @()
    }

# 3. Inside $OUTPUT_NOTEBOOK_PATH (line 73-77) -- new entry:
    "local_pkg" = "/tmp/out_local_pkg.ipynb"

# 4. Inside Build-DockerCmd's switch statement (after the "colab" case) --
#    new branch. The SEED_VER extraction pulls the target version directly
#    from each fixture's own %pip line, so this one branch works correctly
#    for both fixtures without hardcoding a version per notebook. Seeding
#    (from seed_dist, fast plain wheels) makes generation see the package as
#    "installed" so it pins a real, executable DEPENDENCIES entry rather than
#    falling back to a comment-only one; uninstalling before execution forces
#    Cell 2 into a genuine install attempt from dist/ (sdist-only, no wheel).
        "local_pkg" {
            return "pip install --no-cache-dir ipykernel nbconvert==7.17.1 -q && " + `
                   "python -m ipykernel install --user --name python3 && " + `
                   "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/bootstrap pip install --no-cache-dir setuptools wheel && " + `
                   "SEED_VER=`$(grep -oE 'local_test_pkg==[0-9.]+' `"$nb`" | head -1 | cut -d= -f3) && " + `
                   "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/seed_dist pip install --no-cache-dir local_test_pkg==`$SEED_VER && " + `
                   "python notebook_env.py `"$nb`" --output --timeout 2 && " + `
                   "pip uninstall -y local_test_pkg && " + `
                   "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist PIP_NO_BUILD_ISOLATION=1 PIP_NO_CACHE_DIR=1 jupyter nbconvert --to notebook --execute `"$mergedNb`" --output `"/tmp/out_local_pkg.ipynb`" --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        }
