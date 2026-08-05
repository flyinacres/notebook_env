# Notebook Environment-Lock Tool (v29)

A dependency and hardware-requirement scanner for Jupyter/Kaggle notebooks. It looks at what your notebook actually imports, checks that against what's installed in your session, and hands you two cells to paste in so the notebook still runs the same way when someone else opens it.

It's a single, standalone script. Nothing to install, no dependencies of its own — download it, or paste it into a cell, and run it.

## Quick Start

1. Finish your notebook and run it top to bottom in a fresh kernel, so your environment reflects what the notebook actually needs.
2. Get `notebook_env.py` next to your notebook, or paste its contents into a new cell if you're on Kaggle or Colab (see Usage below for which fits your setup).
3. Run it:

   ```bash
   python notebook_env.py your_notebook.ipynb
   ```

4. It prints two blocks. Paste the first into a new Markdown cell and the second into a new code cell, both at the very top of your notebook, above everything else.
5. Save and share the notebook. Whoever opens it next runs your new first two cells before anything else, and their environment gets set up to match yours.

That's the whole workflow. Here's roughly what the two pasted cells look like (shortened):

**Cell 1 (Markdown):**

> ### 🛠️ Environment Setup & Dependency Verification
>
> This notebook includes a pinned environment manifest (`pinned_requirements.txt`) to ensure reproducible execution.
>
> - **Dependency Sync:** Cell 2 will verify your active Python version and apply the exact package manifest recorded by the author.
> - **Hardware Acceleration:** This notebook was created using an active accelerator (`NVIDIA GeForce RTX 3090`, verified via PyTorch). _(only shown if a GPU was actually detected)_

**Cell 2 (Code):**

```python
# =====================================================================
# VERIFIED ENVIRONMENT DEPENDENCIES (2026-08-05 12:00:00)
# =====================================================================
import sys, subprocess

REQUIRED_PYTHON = (3, 11)
# ... checks your Python version matches, warns (doesn't block) on a minor mismatch ...

requirements_content = """# Tested top-level packages for this notebook
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
# cupy (imported as 'cupy' in try/except or conditional block - optional fallback)
"""
# ... writes pinned_requirements.txt and runs `pip install -r` on it ...
```

Notice the last line: `cupy` is commented out because the original notebook only imports it inside a `try/except`, so it's flagged as optional rather than forced on everyone who runs the notebook.

The rest of this document covers the workflow in more depth, batch scanning many notebooks at once, and the honest list of what this tool can't currently do. If you just want to run it once on your own notebook, the steps above are everything you need.

## What this tool checks

- **Imports**: parses your notebook's code via Python's `ast` module to find every top-level import.
- **Installed packages**: correlates each import against your live environment (`pip freeze` plus `importlib.metadata`) to find matching package names and versions. Package name resolution is dynamic — it queries what's actually installed rather than relying solely on a hardcoded map, so it correctly handles cases where the import name differs from the PyPI package name (e.g. `sklearn` → `scikit-learn`) even for packages not explicitly listed in this tool.
- **Guarded / optional imports**: imports found inside `try/except` or `if/else` blocks are tracked separately from unconditional ones. If a package is only ever imported conditionally, its manifest entry is commented out and labeled as an optional dependency rather than pinned as a hard requirement — regardless of whether that package happens to be installed in your own environment. If the same package is _also_ imported unconditionally somewhere else in the notebook, it's treated as required.
- **Dynamic imports (literal form)**: `importlib.import_module("pkg")`, `from importlib import import_module; import_module("pkg")`, aliased forms of both, and bare `__import__("pkg")` are resolved when the argument is a string literal, and treated the same as a normal import. When the argument isn't a literal (e.g. a variable or config value), the tool can't know what's being imported — it emits a diagnostic warning instead of guessing.
- **Optional extras promotion**: if a submodule import matches a package's declared `Provides-Extra` metadata (e.g. `import umap.plot`), the manifest entry is promoted to include that extra (`umap-learn[plot]==...`), with a visible notice printed when this happens.
- **Hardware-tagged builds**: flags packages with build-specific version tags (e.g. `torch==2.3.1+cu121`) and checks whether your notebook already specifies a download index for them.
- **GPU/accelerator usage**: if your notebook imports `torch`, `tensorflow`, or `jax`, checks whether an active GPU/MPS/TPU accelerator was available in your session for that specific framework.

This tool also supports a **batch mode** for scanning many notebooks at once (a course, a team's repo). See Batch Mode below.

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

Scans a directory of notebooks at once, producing an audit-style report rather than per-notebook paste-in blueprints. Useful for an instructor checking a whole course's worth of notebooks, or auditing a shared repo, rather than the single-notebook author workflow above.

```bash
python notebook_env.py --batch ./course_materials
```

This always runs analysis first — it scans every `.ipynb` in the directory, skips non-Python kernels (R, Julia) and notebooks with no language metadata at all, reports parse errors, and summarizes the dependency footprint across the batch (matched packages, missing packages with which notebooks need them, extras promotions, dynamic-import warnings, hardware/index-URL audit). No files are written in this mode.

```bash
python notebook_env.py --batch ./course_materials --universal
```

Writes one `requirements-all.txt` covering every notebook in the batch, correlated against a single shared reference environment (not each notebook's own author environment — batch mode audits against one common target, since aggregating "each notebook's own environment" doesn't mean anything at the directory level).

```bash
python notebook_env.py --batch ./course_materials --output
```

Writes a companion `<name>_merged.ipynb` next to each source notebook, containing the setup cells. The original file is never touched by default. `--in-place` will overwrite the original directly instead (replacing any previously-generated setup cells rather than duplicating them) — use with care, and back up first; this tool does not currently create a backup automatically before an in-place write.

**Known caveat (`--output` mode):** the GPU/accelerator note in a per-notebook generated cell can currently misattribute which framework was actually confirmed to have GPU access, if the host machine has more than one of `torch`/`tensorflow`/`jax` installed. See `DEVELOPMENT.md` for details. Treat the GPU line in `--output` mode as worth double-checking rather than fully trusted, for now.

## Known limitations

- **Dynamic imports with non-literal arguments** (e.g. `importlib.import_module(pkg_name)` where `pkg_name` is a variable) can't be resolved by static AST scanning, since there's no literal module name in the source to read. The tool emits a diagnostic warning in this case rather than guessing. Literal-string dynamic imports (`importlib.import_module("torch")`, including aliased and `from`-imported forms) _are_ detected.
- **Bare relative imports** (`from . import helper`) are silently invisible — nothing is extracted, filtered, or flagged.
- **`exec()`/`eval()`-based code execution** is not inspected — imports embedded in a string passed to `exec()` won't be seen.
- **Package installs introduced via cell magics** aren't harvested yet — `%pip install gdown` (no matching Python import), `%%bash`/`%%sh` shell cells, `%run` external scripts, and `%%writefile`-generated files. Only `--extra-index-url`/`-i` flags are currently picked up from magic/shell lines.
- **Transitive dependencies are not pinned**, only top-level imports. Sub-dependencies can still drift between installs.
- **GPU checks confirm availability, not actual usage** — a GPU can be available and imported without every tensor operation running on it.
- **No Kaggle Docker image tag detection** — no documented environment variable exposes this from inside a running kernel.
- **Path A vs Path B can diverge**: Path A reflects only what's saved to disk; Path B reflects live kernel history, which can be ahead of disk (unsaved edits) or behind it (deleted cells still in `In`). They are not interchangeable views of the same state.
- **No automatic detection of interpreter/kernel mismatch**: Path A correlates against whatever environment is running the script, not necessarily the one the notebook's kernel used. This is not checked or warned about beyond printing the interpreter path — verifying the match is the author's responsibility.
- Hardware-tagged builds (`+cu121`, etc.) are flagged, not auto-corrected — pip's default wheel selection when reinstalling an untagged version is not guaranteed to match the original hardware target.

For the roadmap, in-progress work, and known internal bugs being tracked, see `DEVELOPMENT.md`.
