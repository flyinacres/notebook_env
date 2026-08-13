# Real-World Test Plan — notebook_env.py

Goal: determine whether the tool actually works on real notebooks in real environments — not just whether it runs without crashing. Ordered by cost vs. value, given limited available time. Do each phase fully (or as far as time allows) before moving to the next; don't spread thin across all phases at once.

Log findings as you go:

- Anything that looks like a bug → new entry in `development.md` under "Known bugs," same format as existing entries (what's wrong, where, real-notebook evidence).
- Any import that gets flagged "missing" but obviously shouldn't be → note the import name and correct PyPI name; these get folded into `IMPORT_TO_PYPI_MAP`.
- Any environment/path combination you _didn't_ get to → note it explicitly so it's not silently assumed covered later.

---

## Phase 1 — Environment smoke tests (cheap, do first)

Goal: confirm the tool runs correctly and reports honestly in each environment, before investing time in deeper testing. "Looks right" bar, not exhaustive.

| Environment                            | Path(s) to test                                  | What to specifically check                                                                                                                                                                                                                                                                             | Priority                                                                    |
| -------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Kaggle (GPU notebook)                  | A (CLI) + B (`ne.main()` in a cell)              | `torch.cuda.is_available()` reports correctly against real CUDA; hardware-tag warning fires if the notebook has a `+cu121`-style pin                                                                                                                                                                   | High                                                                        |
| Kaggle (CPU-only notebook)             | A + B                                            | GPU section correctly _absent_ from output (not just wrong, but honestly omitted)                                                                                                                                                                                                                      | High                                                                        |
| Colab                                  | A + B                                            | `google.colab` is tagged as a platform pseudo-module, not reported as a missing package — **not yet confirmed against a real Colab notebook**, only Kaggle/Databricks have real-world confirmation so far                                                                                              | High                                                                        |
| Local Mac (M1)                         | A + B                                            | Torch MPS path (`torch.backends.mps.is_available()`) against real hardware, not a mock — currently only test-mocked, never confirmed on real Apple Silicon; also confirm the "imports torch, no GPU available" honest-negative case                                                                    | High                                                                        |
| GTX 1080 machine (if still accessible) | A + B                                            | Real CUDA detection (`torch.cuda.is_available()` — currently the CI/test environment can't exercise this at all); best chance of a genuine hardware-tagged (`+cu121`) real notebook to confirm that warning against real data, not just synthetic tests                                                | Medium — do if the machine is available, don't go out of your way otherwise |
| Databricks                             | A + B, plus `--batch` if you have a repo of them | If any notebooks are `.py` source-export format (not `.ipynb`) rather than JSON — confirm the tool fails **obviously and clearly**, not silently mis-parses. This format isn't supported (open design question, not a bug), so a confusing failure would be worse than a clean "not supported" message | Medium                                                                      |

**Pass/fail bar for this phase:** for each cell in the table, either (a) output looks correct and matches what you'd expect from reading the notebook, or (b) you've found something wrong and logged it. "I didn't check closely" isn't a pass — note it as untested instead.

---

## Phase 2 — Batch mode across your collections

Run `--batch` (analysis mode, then `--universal`, then `--output`/`--in-place`) across each of your existing notebook collections (course material, personal notebooks, whatever repos you have on hand).

For each collection:

- [ ] Batch analysis report — does the summary look plausible? Any missing-package or hardware-warning entries that look wrong on inspection?
- [ ] `--universal` manifest — spot-check a few packages against what you know is actually needed
- [ ] `--output`/`--in-place` — spot-check a few generated notebooks for correctness (not just "did it run without error")

This phase is mostly about surface-level sanity and catching anything Phase 1's individual-file testing wouldn't reveal (cross-notebook issues, repo-structure issues). Don't spend too long here relative to Phase 3, which is the more targeted version of the same idea.

---

## Phase 3 — Batch-vs-single-file consistency (the highest-value cheap test)

This directly tests whether batch mode and single-file mode agree with each other — and is the fastest way to catch the class of bug this project has run into repeatedly (stale caching, cross-notebook contamination, wrong attribution).

1. **Pick a stratified sample** of 10–15 notebooks from your batch collections — deliberately include:
   - [ ] At least 2–3 GPU notebooks (mix of frameworks if you have them — torch, tensorflow)
   - [ ] At least 1–2 notebooks with local sibling modules (repo-relative imports)
   - [ ] At least 1 notebook with a hardware-tagged package, if you have one
   - [ ] A few plain/boring notebooks as a baseline
2. **Run each sampled notebook individually** via single-file `--output`.
3. **Separately run `--batch --output`** over the _entire_ collection containing those same files.
4. **Diff the generated companion notebooks** for the sampled files between the two runs.
5. **Classify every difference**:
   - Expected and explainable (e.g., batch mode's local-module scoping additionally checks the repo root, which single-file mode doesn't have in the same sense) → note it, move on.
   - Unexplained → this is a real bug. Log it in `development.md` with the specific notebook and the specific diff.

**Pass bar:** every difference between batch and single-file output for the same notebook is either identical or explainable. Zero unexplained diffs.

---

## Phase 4 — End-to-end reproducibility test (expensive, do last, but don't skip)

This is the test `development.md` has flagged as "not yet done" since the project's testing began, and it's the only one that validates the tool's actual claim rather than just its output's plausibility. Everything above checks "does this look right"; this checks "does this actually work."

1. **Pick a small stratified sample** — 5–10 notebooks, given limited time:
   - [ ] 1–2 simple/boring notebooks (low-risk baseline — should just work)
   - [ ] 1 guarded-import-heavy notebook (tests whether the guarded-import messaging is actually sufficient for someone to self-serve)
   - [ ] 1–2 GPU notebooks (best done directly on Kaggle, since that's the only environment where you can validate against a fresh, non-preloaded container)
   - [ ] 1 notebook with a hardware-tagged package, if available
2. **For each notebook**: generate the manifest, then install it into a genuinely **clean/minimal container** — explicitly not a full Kaggle-image container, which would mask an incomplete manifest by already having everything pre-installed.
3. **Run the notebook** against that clean install and confirm it completes.
4. **Record pass/fail per notebook**, and for any failure, whether it's a manifest-generation bug (tool's fault) or a genuinely unfixable gap (e.g., something requiring a platform-specific setup step the tool can't know about).

**Pass bar:** a majority of the sample runs clean off the generated manifest alone. Any failure gets root-caused — tool bug vs. inherent limitation — not just marked "didn't work."

---

## Suggested time allocation (if time is genuinely tight)

1. Phase 1 (smoke tests) — cheap, do fully.
2. Phase 3 (batch-vs-single diff) — moderate cost, highest bug-catching value per hour spent.
3. Phase 4 (reproducibility) — expensive but is the only test that validates the tool's core claim; even a small sample (3–5 notebooks) is worth more than skipping it entirely.
4. Phase 2 (broad batch sanity) — lowest priority if time runs out; Phase 3's sample is a more targeted version of the same idea.
