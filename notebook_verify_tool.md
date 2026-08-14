# Notebook Verification Tool — Future Direction Notes

Captured from a design discussion, for revisit later. Not started. Separate project from `notebook_env.py`; do not let this distract from the current open task list (11 tracked items + structural priorities) until that's closer to resolved.

## What it would be

A local tool that executes a notebook end-to-end (twice) and reports whether it runs, rather than only inferring what _should_ work from static analysis. Complements `notebook_env.py`, which never actually runs anything.

Not hosted. Runs locally, same trust model as opening the notebook in Jupyter yourself. No third-party execution on shared infrastructure.

## Why it's useful

`notebook_env.py`'s output (timeline reconstruction, last-explicit-install-wins, drift audit) is inference from a lossy record — `.ipynb` files don't record true execution history, only a snapshot plus `execution_count`. Actually running the generated cells is the only step that produces empirical ground truth instead of inference. It would also let the "Author Verified Version" label in generated Cell 2 output become accurate rather than aspirational.

## Proposed structure

1. **Run 1 (baseline):** Execute the _original_ notebook, unmodified, in a fresh disposable environment (venv/container). If this fails, stop — not reproducible regardless of anything `notebook_env.py` does. Nothing to attribute.
2. **Run 2 (isolated reconstruction test):** In a _second, separately fresh_ environment, strip or no-op the author's own inline install cells, insert _only_ the `notebook_env.py`-generated setup cells, then run the rest of the notebook body as-is.
   - Isolation matters here specifically: if run 1 and run 2 share an environment or kernel, run 2 "succeeding" could just mean run 1's installs were still present, not that the generated cells were sufficient. Each run needs its own disposable environment or the comparison proves nothing.
3. **Attribution:** Baseline succeeds + reconstruction fails → the timeline/pin reconstruction missed something real — actionable, name the specific package/cell/flag. Both succeed → reconstruction was sufficient to execute without error (see caveat below).
4. Execution/error-capture mechanics likely shouldn't be built from scratch — `papermill` (already in the test corpus) does notebook-execution-with-error-capture already. The new work is the two-environment orchestration plus a correlation layer that ties a run-2 failure back to a specific `notebook_env.py` decision (which package, which scoped flag, which version pin), not the execution harness itself.

## What it still would not resolve

- **"Runs without error" ≠ "produces the same results."** A resolved-but-different transitive version can install cleanly and execute cleanly while silently changing numeric output. This tool would validate reproducibility of _execution_, not of _results_. Worth stating explicitly in any output/docs so a clean run isn't over-trusted.
- **Forward reproducibility only, not historical accuracy.** A successful run proves the generated cells work against _today's_ base image/package availability. It says nothing about whether the timeline reconstruction correctly modeled what the original author actually did back when they wrote it.
- **Doesn't fix pure transitive drift.** Same limitation as the in-notebook drift audit — anything not in `DEPENDENCIES` (never directly imported/installed) isn't tracked, even if this tool surfaces that something broke.
- **Hardware/platform ceiling.** Ron's local hardware (M1 Mac, GTX 1080-class GPU) can't execute CUDA-recent or TPU-dependent notebooks locally regardless of environment isolation — this tool doesn't remove that constraint, it just makes failures on eligible notebooks more informative.
- **Platform pseudo-modules (`dbutils`, `kaggle_secrets`, `google.colab`, etc.) won't resolve locally.** Notebooks depending on Kaggle/Colab/Databricks runtime injection will fail baseline runs locally for reasons unrelated to package reproducibility. Needs a way to distinguish "failed because of environment/package issues" from "failed because this notebook was never meant to run outside its host platform."
- **Cost.** Two full fresh-environment builds + two full notebook executions per analyzed notebook is expensive relative to today's static analysis, especially across a batch corpus. Not free to run casually; likely opt-in (`--verify` flag) rather than default behavior.
- **Execution safety, revisited.** Even without hosting, this still executes arbitrary code from notebooks you didn't write. Local-only removes the third-party/shared-infrastructure risk but not the "this notebook could do something destructive on the machine running it" risk. Disposable/sandboxed environments still matter for that reason alone, not just for isolation-between-runs.

## Sequencing note

This is explicitly a "someday" item, not a next sprint. Current priority order per existing principles: finish resolving the timeline/conflict-detection work from this session, close out the existing 11-item open task list (GPU misattribution, batch mode gaps, local-import false positives, etc.), then consider whether this is worth starting as its own tool.
