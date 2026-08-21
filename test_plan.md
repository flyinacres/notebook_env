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

## Phase 2 — Batch mode across your collections

Run `--batch` (analysis mode, then `--universal`, then `--output`/`--in-place`) across each of your existing notebook collections (course material, personal notebooks, whatever repos you have on hand).

For each collection:

- [ ] Batch analysis report — does the summary look plausible? Any missing-package or hardware-warning entries that look wrong on inspection?
- [ ] `--universal` manifest — spot-check a few packages against what you know is actually needed
- [ ] `--output`/`--in-place` — spot-check a few generated notebooks for correctness (not just "did it run without error")

Mostly surface-level sanity, catching what Phase 1's individual-file testing wouldn't reveal (cross-notebook issues, repo-structure issues). Don't spend too long here relative to Phase 3, which is the more targeted version of the same idea.

---

## Phase 3 — Batch-vs-single-file consistency (highest-value cheap test)

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

## Phase 5 — Automation infrastructure (build once Phases 1–4 have a manual baseline)

Not a substitute for Phases 1–4's human judgment calls (several pass bars above are "does this look right," not just exit-code checks). This phase is about reducing the manual cost of _repeating_ Phases 2–4 over time, once you know by hand what "correct" looks like.

1. **Headless local runner** (nbclient/papermill): script the full loop, append generated Cell 2 to the notebook, execute it, assert exit code and installed versions. This automates Phase 4's mechanics for your local host environment only. It won't catch Kaggle-specific driver behavior (the `cuInit 303` class of bug), only Docker/cloud execution can, so it complements Phase 1/4 rather than replacing them.
2. **Cloud CLI smoke suite** (`kaggle kernels push` against a fixed small set of test notebooks): reserve for occasional pre-release checks, not everyday iteration, given per-run queue/runtime cost. This is the automated version of Phase 1's Kaggle rows, not a replacement for Phase 4's clean-container reproducibility test.

Sequence these after, not instead of, getting through Phase 1 by hand at least once, the automation is only trustworthy once you've verified by eye what its assertions should actually check for.

---

## Suggested time allocation (if time is genuinely tight)

1. Phase 1 (smoke tests) — cheap, do fully. Colab is the biggest current gap.
2. Phase 3 (batch-vs-single diff) — moderate cost, highest bug-catching value per hour spent.
3. Phase 4 (reproducibility) — expensive but validates the tool's core claim; even a small sample (3–5 notebooks) is worth more than skipping it entirely.
4. Phase 2 (broad batch sanity) — lowest priority if time runs out; Phase 3's sample is a more targeted version of the same idea.
5. Phase 5 (automation) — only after 1–4 have given you a manual baseline to trust the automated assertions against.
