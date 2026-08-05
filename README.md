# Notebook Environment-Lock Tool (v29)

A dependency and hardware-requirement scanner for Jupyter/Kaggle notebooks. It scans a notebook's imports, correlates them against the environment you actually ran it in, and generates two paste-in cells (a Markdown explainer and a setup/install cell) so the notebook is reproducible when shared.

This tool does not guarantee bit-for-bit reproducibility. It gives actionable, honest guidance based on what it could actually observe in your session. Version drift in transitive dependencies, and GPU/accelerator behavior, are both explicitly out of scope for guarantees; see Limitations below.

## What this tool checks

- **Imports**: parses your notebook's code via Python's `ast` module to find every top-level import.
- **Installed packages**: correlates each import against your live environment (`pip freeze` plus `importlib.metadata`) to find matching package names and versions. Package name resolution is dynamic — it queries what's actually installed rather than relying solely on a hardcoded map, so it correctly handles cases where the import name differs from the PyPI package name (e.g. `sklearn` → `scikit-learn`) even for packages not explicitly listed in this tool.
- **Guarded / optional imports**: imports found inside `try/except` or `if/else` blocks are tracked separately from unconditional ones. If a package is only ever imported conditionally, its manifest entry is commented out and labeled as an optional dependency rather than pinned as a hard requirement — regardless of whether that package happens to be installed in your own environment. If the same package is _also_ imported unconditionally somewhere else in the notebook, it's treated as required.
- **Dynamic imports (literal form)**: `importlib.import_module("pkg")`, `from importlib import import_module; import_module("pkg")`, aliased forms of both, and bare `__import__("pkg")` are resolved when the argument is a string literal, and treated the same as a normal import. When the argument isn't a literal (e.g. a variable or config value), the tool can't know what's being imported — it emits a diagnostic warning instead of guessing.
- **Optional extras promotion**: if a submodule import matches a package's declared `Provides-Extra` metadata (e.g. `import umap.plot`), the manifest entry is promoted to include that extra (`umap-learn[plot]==...`), with a visible notice printed when this happens.
- **Hardware-tagged builds**: flags packages with build-specific version tags (e.g. `torch==2.3.1+cu121`) and checks whether your notebook already specifies a download index for them.
- **GPU/accelerator usage**: if your notebook imports `torch`, `tensorflow`, or `jax`, checks whether an active GPU/MPS/TPU accelerator was available in your session for that specific framework.

This tool also supports a **batch mode** for scanning many notebooks at once (a course, a team's repo). See Batch Mode below — it's newer and less battle-tested than single-notebook mode; read the Known limitations section before relying on it.

## Required workflow

This tool must be run **by the notebook's author, in the same environment used to develop the notebook**, immediately after testing it. It reads the live state of your session (installed packages, GPU availability) — running it anywhere else, or later, tells you about that environment, not the notebook's actual requirements.

1. Start from a fresh environment/kernel.
2. Run your notebook top to bottom, fixing any errors as they appear.
3. **Restart the kernel** after each fix before rerunning — don't just rerun the failing cell. Previously executed cells stay in memory otherwise, which can hide problems that would break a true cold start.
4. Exercise all code paths, including any GPU-only branches and any guarded (`try/except`) branches, so both the GPU check and the guarded-import detection reflect actual usage rather than partial coverage.
5. Save the notebook.
6. Run this tool (see Usage below).

## Usage

There are two ways to run this tool. Neither is strictly "recommended" over the other — which one fits depends on where you're working.

**Path A — CLI, against a saved `.ipynb` file:**

```bash
python notebook_env.py your_notebook.ipynb
```

Reads the notebook's source directly from the saved `.ipynb` file on disk and runs AST parsing over it. This is the natural fit for local development (VS Code, JupyterLab, PyCharm, a terminal in your project directory) — you save the notebook, then run the script from your shell, without adding any extra code to the notebook itself.

A `.ipynb` file never stores live memory or execution state, on any platform — it only stores whatever cell source was last saved. So Path A reflects exactly what's on disk as of your last save, nothing more and nothing less. If you tested interactively and made further changes without saving, Path A won't see them.

In single-notebook mode, a notebook with no `kernelspec`/`language_info` metadata at all is assumed to be Python rather than rejected — this is a common, valid case for minimal or hand-built notebooks. Metadata is only enforced strictly in batch mode, where it's needed to filter out non-Python kernels (R, Julia) across a whole directory; see Batch Mode below.

**Path B — pasted into a live notebook cell:**

Paste the tool's code into a cell and call `main()`, or adapt it to call `extract_from_active_session()` directly. This reads IPython's `In` history (via `__main__.In`) rather than the saved file, so it reflects what has actually executed in the current kernel session, which can be _ahead of_ disk (recent edits not yet saved) or _behind_ it (cells that ran earlier and were later deleted from the notebook, but are still sitting in `In`). This is generally the better fit for ephemeral cloud runtimes (Colab, Kaggle, remote JupyterHub), where running a separate CLI step against a freshly-saved file is more friction than just running one more cell.

**The stale-memory trap (Path B only):** `__main__.In` accumulates every statement executed for the life of the kernel. If you spent time trying (and then abandoning) `seaborn`, `plotly`, or `bokeh` imports before settling on your final approach, and never restarted the kernel, `In` still contains those abandoned imports even though they're no longer in the notebook. **Restart the kernel and run all cells fresh before the final snapshot** — this flushes the history so only your actual, final code path gets captured.

Do **not** invoke this tool via `import your_module; your_module.main()` inside the notebook you're scanning — the import line itself becomes part of the session history and may show up as a spurious entry in the generated manifest. Paste the tool inline instead.

**Desktop-specific caveat (Path A):** running the CLI correlates imports against whatever Python interpreter runs the script (`pip freeze` under `sys.executable`), not against the notebook's own kernel. On a desktop machine with multiple virtual environments (venv/conda/poetry), it's easy to accidentally run the script from the wrong terminal or the wrong activated environment — the script has no way to detect this, and would silently produce a plausible-looking but wrong manifest rather than an error. **Before running Path A, make sure the environment active in your terminal is the same one the notebook's kernel actually uses.** This is generally less of a concern on Kaggle/Colab, where there's typically only one active kernel environment to begin with.

**Optional flag:**

```bash
python notebook_env.py your_notebook.ipynb --full-freeze
```

Appends a complete `pip freeze` snapshot after the targeted manifest, for cases where you want a full bit-for-bit fallback available. Off by default — a full freeze is generally too much noise for a data scientist audience, and top-level pins are good guidance most of the time.

## Output

The tool prints two blocks to paste into new cells at the top of your notebook:

1. **Markdown cell** — explains the setup, and if applicable, notes hardware-tagged builds and GPU/accelerator usage detected during authoring.
2. **Code cell** — checks the Python version, writes a `pinned_requirements.txt`, and installs it via pip.

Before the blueprint, Path A also prints the active Python interpreter path (`sys.executable`). This exists specifically so you can visually confirm it matches the environment your notebook's kernel actually used — see the desktop caveat above.

Any packages that were only ever imported inside a `try/except` or conditional block appear in the manifest as a commented-out, labeled optional entry rather than a hard pin — even if that package happens to be installed in your own environment. Any dynamic import calls the tool couldn't statically resolve (a variable or expression rather than a string literal) are surfaced as a diagnostic warning rather than silently dropped or guessed at.

## Reading the GPU/accelerator messages

- **"Active accelerator detected"**: a GPU-relevant library was imported and a GPU/MPS/TPU was available and confirmed for that framework in your session. This is a strong signal the notebook needs one, not certainty that every operation used it.
- **"Framework imported, but no active accelerator found"**: you imported a GPU-capable library, but no accelerator was available in this test run. If you intended to require a GPU, this run doesn't confirm that — it just tells you this particular run happened on CPU.
- When multiple GPU frameworks are imported together, only the framework(s) actually confirmed to have an accelerator are reflected in the device name — the message doesn't imply every listed framework had GPU access.

## Batch Mode

Scans a directory of notebooks at once, producing an audit-style report rather than per-notebook paste-in blueprints.

```bash
python notebook_env.py --batch ./course_materials
```

This always runs analysis first — it scans every `.ipynb` in the directory, skips non-Python kernels (R, Julia) and notebooks with no language metadata at all, reports parse errors, and summarizes the dependency footprint across the batch (matched packages, missing packages with which notebooks need them, extras promotions, dynamic-import warnings, hardware/index-URL audit). No files are written in this mode.

Unlike single-notebook mode, batch mode enforces strict language metadata: a notebook with no `kernelspec`/`language_info` at all is treated as unknown and skipped rather than assumed to be Python, since silently assuming Python across an entire directory scan risks misclassifying non-Python or malformed files at scale.

```bash
python notebook_env.py --batch ./course_materials --universal
```

Writes one `requirements-all.txt` covering every notebook in the batch, correlated against a single shared reference environment (not each notebook's own author environment — batch mode audits against one common target, since aggregating "each notebook's own environment" doesn't mean anything at the directory level).

```bash
python notebook_env.py --batch ./course_materials --output
```

Writes a companion `<name>_merged.ipynb` next to each source notebook, containing the setup cells. The original file is never touched by default. `--in-place` will overwrite the original directly instead (replacing any previously-generated setup cells rather than duplicating them) — use with care, and back up first; this tool does not currently create a backup automatically before an in-place write.

**Known caveat (`--output` mode):** GPU/accelerator status is checked once against the host machine for `torch`, `tensorflow`, and `jax` together, then filtered per notebook by which of those frameworks it imports. If more than one of the three frameworks is present on the host but only one of them is actually confirmed to have GPU access, a notebook that imports only one of the _other_ frameworks can still inherit that device's name and "verified via" label in its generated blueprint. Treat the per-notebook GPU note in `--output` mode as a hint to check, not a confirmed per-framework result, until this is tightened to check each framework independently per notebook.

## Known limitations

- **Dynamic imports with non-literal arguments** (e.g. `importlib.import_module(pkg_name)` where `pkg_name` is a variable) can't be resolved by static AST scanning, since there's no literal module name in the source to read. The tool emits a diagnostic warning in this case rather than guessing. Literal-string dynamic imports (`importlib.import_module("torch")`, including aliased and `from`-imported forms) _are_ detected.
- **Bare relative imports** (`from . import helper`) are silently invisible — `node.module` is `None` for this form, so nothing is extracted, filtered, or flagged. Not currently planned; revisit if this turns out to be common in real notebooks scanned during batch testing.
- **`exec()`/`eval()`-based code execution** is not inspected — imports embedded in a string passed to `exec()` won't be seen, and no warning is currently emitted for this case either.
- **Package installs and hardware/toolchain requirements introduced via cell magics** — `%pip`/`%conda` package installs not backed by a matching Python import (e.g. `!pip install gdown`), `%%bash`/`%%sh` shell cells, `%run` external scripts, and `%%writefile`-generated files — are not currently harvested into the manifest. Only `--extra-index-url`/`-i` flags are harvested from magic/shell lines today.
- **Transitive dependencies are not pinned**, only top-level imports. Sub-dependencies can still drift between installs.
- **GPU checks confirm availability, not actual usage** — a GPU can be available and imported without every tensor operation running on it.
- **No Kaggle Docker image tag detection** — no documented environment variable exposes this from inside a running kernel.
- **Path A vs Path B can diverge**: Path A reflects only what's saved to disk; Path B reflects live kernel history, which can be ahead of disk (unsaved edits) or behind it (deleted cells still in `In`). They are not interchangeable views of the same state.
- **No automatic detection of interpreter/kernel mismatch**: Path A correlates against whatever environment is running the script, not necessarily the one the notebook's kernel used. This is not checked or warned about beyond printing the interpreter path — verifying the match is the author's responsibility.
- Hardware-tagged builds (`+cu121`, etc.) are flagged, not auto-corrected — pip's default wheel selection when reinstalling an untagged version is not guaranteed to match the original hardware target.

## Roadmap

- Package as an installable library rather than a standalone script. Note: Path B depends on the tool being paste-able as a single self-contained file into a notebook cell with no install step, which matters specifically on ephemeral, sometimes internet-off runtimes like Kaggle competition rerun mode — any packaging change needs to preserve a single-file distribution form alongside whatever installable form is added, not replace it.
- A separate static-only scanner for notebooks you don't own (no live execution, no environment correlation — file-based AST scan only). Deferred in favor of the current single-notebook and batch-mode work; revisit later.
- Harvest dependencies and tools currently thrown away from cell magics: `%pip`/`%conda` installs without a matching import, `--index-url` (not just `--extra-index-url`), `%%bash`/`%%sh` shell cells, `%run`, and `%%writefile`. These need their own output category rather than folding into the existing import-correlated manifest, since there's no import statement to correlate against for a CLI tool like `gdown`, and conda packages don't reliably map to PyPI names. Sequencing under consideration: index-URL handling first (cheap), then `%pip`/`%conda` package harvesting, then `%%bash` cell classification and `%run`/`%%writefile` only if real notebooks in testing show meaningful use of them.
- Fix the `--output` mode per-notebook GPU misattribution noted above (check each framework's accelerator status independently rather than sharing one host-level result across all three).
- Bare relative imports (`from . import x`) remain silently invisible rather than flagged; low priority unless testing against real notebooks shows this is common.
