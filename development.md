# Development notes

Internal tracking for this tool's own development: known bugs still being fixed, open design questions, and the roadmap. Not needed to just use the tool — see `README.md` for that.

## Known bugs (tracked, not yet fixed)

- **`--output` mode GPU misattribution.** In `apply_output_to_notebook`, GPU/accelerator status is checked once against the host machine for `torch`, `tensorflow`, and `jax` together (`inspect_gpu_environment` is called with all three names regardless of what any given notebook imports), then filtered per notebook by intersecting with that notebook's own imports. `inspect_gpu_environment` returns `device_name`/`active_framework` for whichever one of the three frameworks it happened to confirm GPU access for, but returns `frameworks` as the full intersected set. So a notebook that imports only `tensorflow` (not `torch`) can still inherit the `torch` device name and "verified via PyTorch" label in its generated blueprint, if `torch` was the framework actually confirmed on the host. Fix: check each framework's accelerator status independently and only attach a device/framework label to a notebook if that specific framework was the one confirmed, not just any framework from the shared host-level check.

## Open design questions

- **Full-freeze mode**: should `--full-freeze` stay additive (current behavior — appended after the targeted manifest) or become a replace-mode? Currently additive; not revisited since it was flagged as an open question.
- **Metadata write reliability**: embedding a full freeze directly into `.ipynb` metadata was considered and set aside — companion `.txt` files don't travel with downloaded notebooks, and metadata writes are unreliable against frontend autosave. This blocks any future "single portable file, no companion files" version of full-freeze.
- **Torch/TensorFlow hardware build tag stripping** (`+cu121`, `+cu128`) before pinning: unverified whether this is safe to do automatically. Currently not attempted — tags are flagged, not stripped.

## Roadmap

- **Package as an installable library**, rather than only a standalone script. Constraint to preserve: Path A/B currently both depend on the tool being usable as a single self-contained file with no install step — Path B specifically depends on being paste-able into a notebook cell, which matters on ephemeral, sometimes internet-off runtimes (e.g. Kaggle competition rerun mode). Any packaging change needs to keep a single-file distribution form available alongside whatever installable form is added, not replace it.
- **A separate static-only scanner** for notebooks you don't own (no live execution, no environment correlation — file-based AST scan only). Deferred in favor of the current single-notebook and batch-mode work.
- **Harvest dependencies and tools currently thrown away from cell magics**: `%pip`/`%conda` installs without a matching import (e.g. `gdown`), `--index-url` (not just `--extra-index-url`), `%%bash`/`%%sh` shell cells, `%run`, and `%%writefile`. These need their own output category rather than folding into the existing import-correlated manifest — there's no import statement to correlate a CLI-only tool like `gdown` against, and conda package names don't reliably map to PyPI names, so conda installs shouldn't be checked against `pip freeze` the way pip installs are. Suggested sequencing:
  1. `--index-url` harvesting (cheap, extends existing regex scan).
  2. `%pip`/`%conda` package name harvesting into a distinct "auxiliary tools" section, with `-r requirements.txt` handled as a warning (file path resolution isn't guaranteed to match) rather than silently followed.
  3. `%%bash`/`%%sh` cell classification (requires branching the extraction pipeline on cell type before deciding AST-parse vs. line-scan) and `%run`/`%%writefile` handling — only worth building if real notebooks (Kaggle sample, course material) actually show meaningful use of these; check via a quick regex frequency sweep before implementing.
- **Fix the `--output` mode GPU misattribution** noted above.
- **Bare relative imports** (`from . import x`) remain silently invisible rather than flagged. Low priority — revisit only if real-notebook testing shows this pattern is common.
- **`exec()`/`eval()` diagnostic warning**: cheap to add (flag any `exec(`/`eval(` call site as a diagnostic, without attempting to parse the string argument), not yet implemented.

## Testing notes

- `kitchen_sink.ipynb` (in `fixtures/`) is the primary AST edge-case fixture: name mismatches, dotted/submodule imports, guarded imports, dynamic imports (literal and non-literal), star imports, commented-out and string-literal false positives, a hardware-tagged package with a matching index URL, and a bare relative import.
- Test suite is split by concern: `test_notebook_env.py` (unit-level AST/resolution/blueprint tests), `test_notebook_env_fixtures.py` (fixture-based end-to-end), `test_batch_mode.py` (batch orchestration happy paths), `test_batch_failures.py` (batch failure modes — corrupted JSON, missing `cells`, hidden/venv directory skipping, non-Python kernel filtering).
- Full suite should be run and passing before sharing any change — this has been the practice throughout and is worth keeping as changes accumulate.
