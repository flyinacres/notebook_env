# Real-World Test Plan — notebook_env.py

Goal: determine whether the tool actually works on real notebooks in real environments, not just whether it runs without crashing. Ordered by cost vs. value. Do each phase fully (or as far as time allows) before moving to the next; don't spread thin across all phases at once.

Log findings as you go:

- Anything that looks like a bug → new entry in `development.md` under "Known bugs," same format as existing entries (what's wrong, where, real-notebook evidence).
- Any import that gets flagged "missing" but obviously shouldn't be → note the import name and correct PyPI name; these get folded into `IMPORT_TO_PYPI_MAP`.
- Any environment/path combination you _didn't_ get to → note it explicitly so it's not silently assumed covered later.

Known open issue, unrelated to this plan: `test_atomic_last_wins_replaces_all_fields_indivisibly` currently fails (fires 2 conflict warnings instead of 1). This is the index-url conflict detection gap already tracked in `development.md`, not something to chase during environment testing.

---

## Phase 0 — Interactive-session mechanics (done)

These don't depend on any cloud environment and are now covered by regression tests, added after the Kaggle paste-run session surfaced two bugs that Phases 1–4 below would never have caught (they test analysis correctness, not kernel-session mechanics):

- [x] `sys.argv` contamination from `ipykernel_launcher.py -f <connection.json>` incorrectly populating `args.notebook` and hijacking Path A instead of Path B — `test_argv_contamination_from_ipykernel_launcher_clears_notebook_arg`
- [x] Duplicate stderr log handlers from re-running the module in the same live kernel — `test_logger_handler_configuration_prevents_duplicate_logging`
- [x] `__main__.In` history extraction correctly filters out notebook_env's own source/invocation cells — `test_live_kernel_history_self_introspection_filter`

Keep this class of test in mind going forward: any bug discovered via manual paste-and-run in Phase 1 that turns out to be a kernel/session mechanic (not a package or hardware correctness issue) belongs here, as a cheap local regression test, before it belongs in Phase 1's environment matrix.

---

## Phase 1 — Environment smoke tests (cheap, do first)

Goal: confirm the tool runs correctly and reports honestly in each environment, before investing time in deeper testing. "Looks right" bar, not exhaustive. For each row, both A (CLI on a saved file) and B (live paste-run, `ne.main()` in a cell) should be checked — B is what actually caught the Phase 0 bugs, don't skip it in favor of A alone.

| Environment                            | Path(s)                                          | What to specifically check                                                                                                                                                                                                                   | Priority                                           |
| -------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Kaggle (GPU notebook)                  | A + B                                            | `torch.cuda.is_available()` reports correctly against real CUDA; hardware-tag warning fires if the notebook has a `+cu121`-style pin                                                                                                         | High                                               |
| Kaggle (CPU-only notebook)             | A + B                                            | GPU section correctly _absent_ from output (honest omission, not silently wrong); confirm no repeat of the argv/logger issues on a second paste-run in the same kernel                                                                       | High                                               |
| Colab                                  | A + B                                            | `google.colab` tagged as a platform pseudo-module, not reported as missing — **not yet confirmed on a real Colab notebook**, only Kaggle/Databricks have real-world confirmation so far. This is the biggest untested gap right now.         | High                                               |
| Local Mac (M1)                         | A + B                                            | Torch MPS path (`torch.backends.mps.is_available()`) against real hardware, not a mock — currently only test-mocked; also confirm the "imports torch, no GPU available" honest-negative case                                                 | High                                               |
| GTX 1080 machine (if still accessible) | A + B                                            | Real CUDA detection (CI/test environment can't exercise this at all); best chance of a genuine hardware-tagged (`+cu121`) real notebook to confirm that warning against real data                                                            | Medium — do if available, don't go out of your way |
| Databricks                             | A + B, plus `--batch` if you have a repo of them | If any notebooks are `.py` source-export format (not `.ipynb`) — confirm the tool fails **obviously and clearly**, not silently mis-parses. Not supported by design; a confusing failure would be worse than a clean "not supported" message | Medium                                             |

**Pass/fail bar:** for each cell, either (a) output looks correct and matches what you'd expect from reading the notebook, or (b) you've found something wrong and logged it. "I didn't check closely" isn't a pass, note it as untested instead.

**Update:** the GPU/Kaggle/Mac rows above test whether real hardware is _present_; they don't test whether the tool's detection and code-generation logic is _correct_, which turns out not to need real hardware at all — see Phase 5f. Real-hardware confirmation above is still valuable (it's the only way to test the actual detection call against real CUDA/MPS/TPU), but it's no longer the only way to exercise this code path, and Phase 5f can run today without waiting on Kaggle GPU quota or Mac access.

---

## Phase 2 — Batch mode across your collections (substantially run)

Run `--batch` (analysis mode, then `--universal`, then `--output`/`--in-place`) across each of your existing notebook collections (course material, personal notebooks, whatever repos you have on hand).

For each collection:

- [x] Batch analysis report — run across the full 130-notebook corpus plus targeted subdirectories (`big_data`, `databricks`, `datashader`). Surfaced several real bugs (GPU probe crash, harvested-name duplication, false-positive local names, `pip` self-reference, `python-dotenv` annotation inconsistency) — see `development.md`, most now fixed.
- [x] `--universal` manifest — spot-checked, same findings as above (the manifest and console report share the underlying duplication bug, now fixed).
- [x] `--output`/`--in-place` — spot-checked across `big_data`; surfaced the non-idempotent-write bug (now fixed) and the directory-pollution problem that led to `--output-dir` being built.

Mostly surface-level sanity, catching what Phase 1's individual-file testing wouldn't reveal (cross-notebook issues, repo-structure issues). This phase did most of the actual bug-finding work this round, more than Phase 3 below managed to, since Phase 3's diff got sidetracked into the idempotency bug before completing a clean comparison.

---

## Phase 3 — Batch-vs-single-file consistency (attempted, not completed as designed)

Tests whether batch mode and single-file mode agree with each other, the fastest way to catch the class of bug this project has run into repeatedly (stale caching, cross-notebook contamination, wrong attribution).

1. **Pick a stratified sample** of 10–15 notebooks from your batch collections:
   - [ ] At least 2–3 GPU notebooks (mix of frameworks if available — torch, tensorflow)
   - [ ] At least 1–2 notebooks with local sibling modules (repo-relative imports)
   - [ ] At least 1 notebook with a hardware-tagged package, if you have one
   - [ ] A few plain/boring notebooks as a baseline
2. **Run each sampled notebook individually** via single-file `--output`.
3. **Separately run `--batch --output`** over the entire collection containing those files.
4. **Diff the generated companion notebooks** for the sampled files between the two runs.
5. **Classify every difference**: expected/explainable → note and move on; unexplained → real bug, log it with the specific notebook and diff.

**What actually happened**: a first attempt at step 3–4 (running single-file `--output` against an already-merged `model_training_merged.ipynb` rather than the original) surfaced the `apply_output_to_notebook` idempotency bug directly, valuable, but this wasn't the clean-diff comparison the phase was designed to produce, and the stratified sample above was never actually assembled or run properly. **Still genuinely open.** Given Phase 5c below is building a reusable version of this exact test, it may make more sense to do this phase via that automation once built, rather than repeating it by hand now — your call.

**Pass bar:** every difference is identical or explainable. Zero unexplained diffs.

---

## Phase 4 — End-to-end reproducibility test (expensive, do last, don't skip)

This is the test that validates the tool's actual claim rather than just its output's plausibility. Everything above checks "does this look right"; this checks "does the generated Cell 2 actually produce a working environment." Confirm each notebook's full loop: generate → paste/append Cell 2 → **execute Cell 2** → confirm the notebook runs. Generating a plausible-looking Cell 2 isn't a pass by itself.

1. **Pick a small stratified sample** — 5–10 notebooks:
   - [ ] 1–2 simple/boring notebooks (low-risk baseline)
   - [ ] 1 guarded-import-heavy notebook (tests whether the guarded-import messaging is sufficient for someone to self-serve)
   - [ ] 1–2 GPU notebooks (best done directly on Kaggle, the only environment where you can validate against a fresh, non-preloaded container)
   - [ ] 1 notebook with a hardware-tagged package, if available
2. **For each notebook**: generate the manifest, install it into a genuinely **clean/minimal container**, explicitly not a full Kaggle-image container, which would mask an incomplete manifest by already having everything pre-installed.
3. **Run the notebook** against that clean install and confirm it completes.
4. **Record pass/fail per notebook**; for failures, root-cause as manifest-generation bug vs. genuinely unfixable gap (e.g. a platform-specific setup step the tool can't know about).

**Pass bar:** a majority of the sample runs clean off the generated manifest alone. Every failure gets root-caused, not just marked "didn't work."

---

## Phase 4.5 — Structural fixture testing (done by hand, ready to automate)

Separate from the organic corpus above: four purpose-built directory structures (`build_test_structures.py`) targeting specific structural questions the organic corpus doesn't reliably exercise — subdirectory helpers (both package-style and `sys.path.append`-style), root-level helper resolution, `--output-dir` duplicate-stem collision avoidance, and relative-asset mirroring. All four run and checked by hand; see `development.md` for exact findings (three confirmed working as designed or fixed, one — `sys.path.append` — confirmed real but narrow and deliberately deferred). Ready to convert into pytest fixtures, this is Phase 5b below, since what "correct" means for each case is already fully decided.

---

## Phase 5 — Automation infrastructure

Not a substitute for Phases 1–4's human judgment calls, several pass bars there are "does this look right," not just exit-code checks. This phase exists because hand-testing has already found real bugs the unit suite couldn't (argv contamination, duplicate handlers, non-idempotent writes, unmerged package names across code paths), confirming the unit suite alone isn't sufficient, but hand-testing itself doesn't scale as the primary ongoing method going forward. Goal: convert what's already been verified by hand into something that runs unattended and catches regressions automatically.

### 5a — `--format json` output mode (new scope, dual-purpose)

Design and build a machine-readable output mode, generated from the _same_ internal report structure the console output already builds, not a second independent computation. This is the single most important constraint on this item: this project has hit the "two code paths compute the same answer slightly differently and drift apart" bug more than once this session (GPU attribution, harvested-name normalization, the skip-suffix/managed-metadata inconsistency), and a JSON serializer built as a separate pass over the same data would be the same failure mode again. One report object, two renderers.

This is genuinely dual-purpose, not just test scaffolding:

- **Test infra**: every downstream automation piece (structural fixtures, batch-vs-single diff, corpus goldfiles) can assert on parsed fields instead of string-matching formatted console text with emoji and wrapped prose, far less brittle to cosmetic wording changes.
- **Real product feature**: gives users a way to wire this into CI/release automation without parsing human-formatted output, worth documenting in README once it exists, not just `development.md`.

Not blocking — build in parallel with 5b below, since 5b's fixtures are already fully specified against current output and don't need to wait.

### 5b — Structural fixtures → pytest

Convert `build_test_structures.py`'s four cases (see Phase 4.5) into `tmp_path`-based pytest fixtures: build structure → run CLI via `subprocess.run` → assert on output. Already fully specified since you validated by hand exactly what each case should assert — mechanical, not exploratory, at this point. Cheapest item in this phase, do first regardless of 5a's progress.

### 5c — Batch-vs-single-file diff helper

Generalize Phase 3's diff test (attempted, not completed by hand — see above) into a reusable `assert_batch_matches_single_file(fixture_dir)` function: run batch `--output` and single-file `--output` over the same notebook, diff the results, fail on unexplained differences. Once 5a exists, this should diff parsed JSON structures rather than raw notebook-cell text — semantic diffing is easier to classify programmatically (which field changed) than a raw text diff. This is also the natural way to finally close out Phase 3 properly, rather than repeating it by hand.

### 5d — Idempotency harness

Generalize what's been checked by hand for `--output`/`--in-place`/`--output-dir`: run each mode twice against the same fixture, assert exactly one managed cell survives, no stacking. Directly encodes the idempotency bug found and fixed this session, so it's also the harness's own first regression test.

### 5e — Corpus goldfile testing (real 130-notebook corpus)

Snapshot-testing pattern, not a hardcoded-assertion pattern: save known-good output (ideally JSON, once 5a exists) as goldfiles, diff future runs against them, flag any difference for review rather than auto-failing or auto-passing. When a change is intentional (a fix like the normalization bug, or `IMPORT_TO_PYPI_MAP` growing), regenerate the goldfile and review that diff like any other code change before committing it, the same discipline already used for reviewing real code changes in this project. When a difference shows up that wasn't expected from anything you changed, that's a regression, not a goldfile update. Separate from the pytest suite proper given the corpus's size and gitignored status — run periodically (pre-release, or on demand), not on every commit.

### Sabotage-testing the harness itself

Before trusting any of 5b–5e as a safety net, deliberately reintroduce a fixed bug (the argv contamination or the idempotency bug are good candidates, both well-understood) and confirm the relevant test actually fails. `development.md` already documents doing exactly this for the memoize decorator; worth the same discipline here given how much of this session was "the existing suite didn't catch it."

### Suggested build order

1. **5b** (structural fixtures) — cheapest, fully specified, no dependencies.
2. **5a** (`--format json`) — design and build in parallel with 5b; unblocks better versions of 5c/5e.
3. **5g** (live-kernel automation) — new addition, but has proven historical bug yield (Phase 0's three bugs); worth prioritizing above despite being newly scoped.
4. **5d** (idempotency harness) — straightforward once 5b's fixture-building pattern exists.
5. **5c** (batch-vs-single diff) — build against text output if 5a isn't ready yet; migrate to JSON once it is. Also finally closes out Phase 3.
6. **5f** (hardware/accelerator mocking) — new addition, no dependencies on the others, can run in parallel with any of the above.
7. **5h** (conda / network-restricted / read-only / encoding) — new addition, no dependencies, sequence relative to actual user-base risk.
8. **5e** (corpus goldfiles) — last, benefits most from 5a existing first, and is the least urgent to run frequently.

Deviate from this order if something learned along the way argues for it — this is a starting sequence, not a commitment.

### 5f — Hardware/accelerator mocking in Docker (new, no real hardware needed)

Discovery this session: `inspect_gpu_environment` (and the per-framework `probe_torch_gpu`/`probe_tensorflow_gpu`/`probe_jax_gpu` functions) run entirely at **generation time**, and their result is baked into a **static markdown section** in Cell 1 — not a live check embedded in the generated executable Cell 2. This means the code path that matters (detect hardware → correctly document it) can be fully exercised without any real GPU/TPU/MPS at all. We only need the exact library calls each probe makes to return "found," which is a small, precise surface:

- **torch**: `torch.cuda.is_available()` → `True` plus `torch.cuda.get_device_name(0)` → a string (CUDA path), or `torch.backends.mps.is_available()` → `True` (Apple Silicon MPS path)
- **tensorflow**: `tf.config.list_physical_devices('GPU')` → non-empty list, `tf.config.experimental.get_device_details(...)` → a dict with `device_name`
- **jax**: `jax.devices()` → objects with `.platform` in `("gpu", "tpu", "metal")` and a `.device_kind`

Each is fakeable with a tiny stub package a few lines long — no multi-GB real ML library installs needed. One wrinkle: the notebook must actually `import <framework>` for `inspect_gpu_environment` to probe it at all (gated by `SUPPORTED_GPU_FRAMEWORKS.intersection(expanded_imports)`), and for the framework to get a real `DEPENDENCIES` entry (rather than falling into the comment-only fallback for "not currently installed" packages — see Phase 7 below), the stub needs real package metadata, not just an importable `.py` file: `importlib.metadata.version('torch')` must resolve, so build it as an actual trivial wheel or `pip install -e` it, not just drop it on `PYTHONPATH`.

This effectively gives Docker-based e2e coverage of **CUDA, Apple Silicon MPS, TensorFlow GPU, and JAX GPU/TPU/Metal detection** — every hardware permutation Phase 1 currently marks "needs real hardware" — as a code-generation-correctness test, distinct from (and much cheaper than) actually running compute on that hardware.

**Suggested first step:** build one stub (fake `torch`, CUDA path) as a proof of concept, confirm it flows through to the generated `gpu_markdown_section` text, before generalizing to MPS/TensorFlow/JAX.

**Still needs real hardware:** the actual correctness of `torch.cuda.is_available()` itself against real silicon — this only tests that _notebook_env.py_ does the right thing _given_ a hardware signal, not that the signal-producing libraries are right. That's still Phase 1's job, just no longer the only way to exercise this code.

### 5g — Live-kernel / interactive-session automation (new, highest proven value)

Every e2e fixture built so far (Phase 7 below included) uses `jupyter nbconvert --execute`, which only performs clean, linear, one-shot execution. Real usage isn't linear: run a cell, edit it, rerun out of order, restart the kernel, rerun a single cell. Phase 0 already found three real bugs this exact way (argv contamination, duplicate log handlers, kernel-history self-introspection) — bugs that no amount of nbconvert-based e2e testing could ever catch, because nbconvert structurally can't represent "the same kernel already ran this once." This is the single gap on this list with _proven_ historical bug yield, currently only exercised by hand.

This is fully automatable, no hardware needed: use `jupyter_client`'s `KernelManager`/`BlockingKernelClient` to start a real kernel inside the container and drive it via the Jupyter messaging protocol directly — send `execute_request` messages in whatever order the test wants (rerun cell 2 twice, run cell 3 before cell 1, restart and rerun only cell 2), inspect `iopub` messages for outputs/errors. This converts Phase 0's manual-paste-and-run discovery method into a repeatable regression harness, rather than relying on catching this class of bug by hand again in the future.

**Priority:** given the proven bug yield, this should be sequenced ahead of 5c/5e in practice, even though it's new scope not in the original build order.

### 5h — Other Docker-mockable environment conditions (new, no hardware needed)

None of these need real hardware, all are currently untested by anything, and none appear in Phase 1's environment matrix (which is entirely hardware/platform-focused):

- **Conda-managed environments.** `CONDA_INSTALL_PATTERN` already exists in `notebook_env.py` as real logic (detects `%conda install` lines, warns they're untracked in pip manifests) but has zero test coverage at any level found so far. Pip-installing into a conda env is a well-known real landmine (ABI mismatches on compiled packages like numpy/scipy) and conda is extremely common in this tool's actual target audience (data scientists, students). A miniconda-base-image Docker fixture is straightforward to build.
- **Network-restricted / air-gapped environments.** `docker run --network none`, or a proxy container, tests whether the tool degrades with a clear diagnostic when pip genuinely can't reach PyPI — real scenario (Kaggle no-internet competition mode, corporate firewalls), not exercised anywhere currently.
- **Read-only source filesystem.** `docker run --read-only` (or a chmod'd mount) tests whether `--in-place`/`--output`/`--output-dir` fail with a clear, actionable error against a read-only source, rather than a confusing crash.
- **Encoding/line-ending edge cases.** CRLF notebooks from Windows editors, non-UTF-8 residue in cell source (rare but real from copy-paste), unicode filenames/paths. Given this tool's actual dev environment is Windows/WSL2 and its target audience spans OSes, this is a plausible and currently-unexercised bug source in the AST/regex-based source scanning.

---

## Phase 6 — Still-valid original automation ideas, now sequenced after Phase 5

These were the original automation ideas for this plan; still worth doing, just no longer the immediate next step given Phase 5's higher-leverage infra work above.

1. **Headless local runner** (nbclient/papermill): script the full loop, append generated Cell 2 to the notebook, execute it, assert exit code and installed versions. Automates Phase 4's mechanics for your local host environment only. Won't catch Kaggle-specific driver behavior (the `cuInit 303` class of bug), only Docker/cloud execution can, so it complements Phase 1/4 rather than replacing them.
2. **Cloud CLI smoke suite** (`kaggle kernels push` against a fixed small set of test notebooks): reserve for occasional pre-release checks, not everyday iteration, given per-run queue/runtime cost. Automated version of Phase 1's Kaggle rows, not a replacement for Phase 4's clean-container reproducibility test.

---

## Phase 7 — E2E install-engine correctness under partial failure (completed this session)

New sub-area, not originally called out in this plan: does the sequential per-package install engine actually behave correctly when one specifier in a multi-package manifest fails, as opposed to just documenting that it's supposed to (the original motivation for building it sequential rather than atomic in the first place).

- [x] **Discovered and fixed a fixture-design bug before it shipped**: `DEPENDENCIES` only ever contains packages already installed at generation time (`resolve_pypi_package_and_extras` demotes anything not currently installed to an informational comment, regardless of import or explicit pip pin — confirmed against source, not assumed). This means a genuinely-nonexistent package can _never_ reach the sequential installer; the original `test_e2e_partial_install_failure` fixture was actually testing "an uncaught Python import crashes a notebook," true of any code, not anything specific to this tool. Retired; the ground it thought it covered (informational-comment generation for uninstalled imports) was already covered twice over by existing unit tests (`test_uninstalled_package_produces_fallback_comment_in_main`, `test_uninstalled_auxiliary_tools_rendered_as_unpinned_comment`).
- [x] `test_partial_install_recovery.ipynb` (positive) — an already-installed real package re-pinned to a genuinely bad version fails via the engine, while a sibling already-installed package installs cleanly; both the failure diagnostic and the success message get verified against the executed notebook's actual content (not terminal stdout — nbconvert only ever streams per-cell `print()` output into the output notebook's JSON, never to the container's own stdout/stderr).
- [x] `test_e2e_failed_repin_surfaces_downstream.ipynb` (negative) — proves a silently-failed re-pin surfaces as a clear, diagnosable downstream error (not silent wrong-version behavior) when code actually depends on the re-pin having succeeded.
- [x] `test_numpy_old_pin_preserves_api.ipynb` (positive) — proves _correct_ pinning preserves old, working behavior across a real, documented API break (`numpy.bool` alias, removed in 1.24). **Sabotage-tested**: confirmed to actually fail (not vacuously pass) when the pin is dropped end-to-end (both the notebook's `%pip install` line and the environment bootstrap line), and confirmed to pass again once reverted.
- [x] **Structural negative-fixture verification**, replacing a single-substring traceback grep: `--allow-errors` makes nbconvert always write the output notebook regardless of outcome; `tests/runners/check_negative_fixture.py` parses the notebook JSON directly and asserts exactly one cell error occurred, with the expected `ename` and an `evalue` substring — catches "wrong failure occurred" in a way a traceback-substring match structurally cannot. Pulled out of inline PowerShell (hit real nested-quoting/escaping failures passing complex strings from PowerShell to a containerized `bash -c`) into a standalone, independently-testable script.
- [ ] Same negative-fixture coverage does not yet exist for the `kaggle`/`colab` tiers — lower priority than it sounds, since the install engine is shared code across all three tiers (a Kaggle/Colab run mostly re-exercises environment differences: system-site-packages venv, kernel setup — not new engine logic).

---

## Suggested time allocation (if time is genuinely tight)

1. **Phase 5b** (structural fixtures → pytest) — cheapest automation win, already fully specified, start here.
2. **Phase 1** (smoke tests) — cheap, do fully by hand where automation doesn't yet cover it. Colab is the biggest current gap.
3. **Phase 5a** (`--format json`) — build in parallel with the above; unblocks everything downstream in Phase 5.
4. **Phase 5g** (live-kernel automation) — new, but the only item anywhere in this plan with _proven_ historical bug yield (Phase 0's three bugs). Worth pulling forward ahead of 5c/5d despite being newly added.
5. **Phase 5c/5d** (diff + idempotency harnesses) — moderate cost, highest ongoing bug-catching value per hour invested, and closes out Phase 3 properly.
6. **Phase 5f** (hardware/accelerator mocking) — new, no real hardware needed, closes most of Phase 1's GPU/MPS/TPU gaps at the code-generation-correctness level; real-hardware confirmation in Phase 1 still separately valuable.
7. **Phase 4** (reproducibility) — expensive but validates the tool's core claim; even a small sample (3–5 notebooks) is worth more than skipping it entirely.
8. **Phase 5h** (conda / network-restricted / read-only / encoding) — no hardware needed, currently zero coverage anywhere; sequence relative to your actual user base's likely environment mix.
9. **Phase 5e** (corpus goldfiles) and **Phase 6** (headless/cloud automation) — lowest immediate priority; both benefit from everything above existing first.
