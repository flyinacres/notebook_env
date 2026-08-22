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
3. **5d** (idempotency harness) — straightforward once 5b's fixture-building pattern exists.
4. **5c** (batch-vs-single diff) — build against text output if 5a isn't ready yet; migrate to JSON once it is. Also finally closes out Phase 3.
5. **5e** (corpus goldfiles) — last, benefits most from 5a existing first, and is the least urgent to run frequently.

Deviate from this order if something learned along the way argues for it — this is a starting sequence, not a commitment.

---

## Phase 6 — Still-valid original automation ideas, now sequenced after Phase 5

These were the original automation ideas for this plan; still worth doing, just no longer the immediate next step given Phase 5's higher-leverage infra work above.

1. **Headless local runner** (nbclient/papermill): script the full loop, append generated Cell 2 to the notebook, execute it, assert exit code and installed versions. Automates Phase 4's mechanics for your local host environment only. Won't catch Kaggle-specific driver behavior (the `cuInit 303` class of bug), only Docker/cloud execution can, so it complements Phase 1/4 rather than replacing them.
2. **Cloud CLI smoke suite** (`kaggle kernels push` against a fixed small set of test notebooks): reserve for occasional pre-release checks, not everyday iteration, given per-run queue/runtime cost. Automated version of Phase 1's Kaggle rows, not a replacement for Phase 4's clean-container reproducibility test.

---

## Suggested time allocation (if time is genuinely tight)

1. **Phase 5b** (structural fixtures → pytest) — cheapest automation win, already fully specified, start here.
2. **Phase 1** (smoke tests) — cheap, do fully by hand where automation doesn't yet cover it. Colab is the biggest current gap.
3. **Phase 5a** (`--format json`) — build in parallel with the above; unblocks everything downstream in Phase 5.
4. **Phase 5c/5d** (diff + idempotency harnesses) — moderate cost, highest ongoing bug-catching value per hour invested, and closes out Phase 3 properly.
5. **Phase 4** (reproducibility) — expensive but validates the tool's core claim; even a small sample (3–5 notebooks) is worth more than skipping it entirely.
6. **Phase 5e** (corpus goldfiles) and **Phase 6** (headless/cloud automation) — lowest immediate priority; both benefit from everything above existing first.
