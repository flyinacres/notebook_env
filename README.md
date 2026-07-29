# Notebook Environment-Lock Tool (v19)

A dependency and hardware-requirement scanner for Jupyter/Kaggle notebooks. It scans a notebook's imports, correlates them against the environment you actually ran it in, and generates two paste-in cells (a Markdown explainer and a setup/install cell) so the notebook is reproducible when shared.

This tool does not guarantee bit-for-bit reproducibility. It gives actionable, honest guidance based on what it could actually observe in your session. Version drift in transitive dependencies, and GPU/accelerator behavior, are both explicitly out of scope for guarantees; see Limitations below.

## What this tool checks

- **Imports**: parses your notebook's code via Python's `ast` module to find every top-level import.
- **Installed packages**: correlates each import against your live environment (`pip freeze`) to find matching package names and versions.
- **Hardware-tagged builds**: flags packages with build-specific version tags (e.g. `torch==2.3.1+cu121`) and checks whether your notebook already specifies a download index for them.
- **GPU/accelerator usage**: if your notebook imports `torch`, `tensorflow`, or `jax`, checks whether an active GPU/MPS/TPU accelerator was available in your session for that specific framework.

## Required workflow

This tool must be run **by the notebook's author, in the same environment used to develop the notebook**, immediately after testing it. It reads the live state of your session (installed packages, GPU availability) — running it anywhere else, or later, tells you about that environment, not the notebook's actual requirements.

1. Start from a fresh environment/kernel.
2. Run your notebook top to bottom, fixing any errors as they appear.
3. **Restart the kernel** after each fix before rerunning — don't just rerun the failing cell. Previously executed cells stay in memory otherwise, which can hide problems that would break a true cold start.
4. Exercise all code paths, including any GPU-only branches, so the GPU check reflects actual usage rather than partial coverage.
5. Save the notebook.
6. Run this tool (see Usage below).

## Usage

There are two ways to run this tool. Neither is strictly "recommended" over the other — which one fits depends on where you're working.

**Path A — CLI, against a saved `.ipynb` file:**

```bash
python env_lock.py your_notebook.ipynb
```

Reads the notebook's source directly from the saved `.ipynb` file on disk and runs AST parsing over it. This is the natural fit for local development (VS Code, JupyterLab, PyCharm, a terminal in your project directory) — you save the notebook, then run the script from your shell, without adding any extra code to the notebook itself.

A `.ipynb` file never stores live memory or execution state, on any platform — it only stores whatever cell source was last saved. So Path A reflects exactly what's on disk as of your last save, nothing more and nothing less. If you tested interactively and made further changes without saving, Path A won't see them.

**Path B — pasted into a live notebook cell:**

Paste the tool's code into a cell and call `main()`, or adapt it to call `extract_from_active_session()` directly. This reads IPython's `In` history (via `__main__.In`) rather than the saved file, so it reflects what has actually executed in the current kernel session, which can be _ahead of_ disk (recent edits not yet saved) or _behind_ it (cells that ran earlier and were later deleted from the notebook, but are still sitting in `In`). This is generally the better fit for ephemeral cloud runtimes (Colab, Kaggle, remote JupyterHub), where running a separate CLI step against a freshly-saved file is more friction than just running one more cell.

**The stale-memory trap (Path B only):** `__main__.In` accumulates every statement executed for the life of the kernel. If you spent time trying (and then abandoning) `seaborn`, `plotly`, or `bokeh` imports before settling on your final approach, and never restarted the kernel, `In` still contains those abandoned imports even though they're no longer in the notebook. **Restart the kernel and run all cells fresh before the final snapshot** — this flushes the history so only your actual, final code path gets captured.

Do **not** invoke this tool via `import your_module; your_module.main()` inside the notebook you're scanning — the import line itself becomes part of the session history and may show up as a spurious entry in the generated manifest. Paste the tool inline instead.

**Desktop-specific caveat (Path A):** running the CLI correlates imports against whatever Python interpreter runs the script (`pip freeze` under `sys.executable`), not against the notebook's own kernel. On a desktop machine with multiple virtual environments (venv/conda/poetry), it's easy to accidentally run the script from the wrong terminal or the wrong activated environment — the script has no way to detect this, and would silently produce a plausible-looking but wrong manifest rather than an error. **Before running Path A, make sure the environment active in your terminal is the same one the notebook's kernel actually uses.** This is generally less of a concern on Kaggle/Colab, where there's typically only one active kernel environment to begin with.

**Optional flag:**

```bash
python env_lock.py your_notebook.ipynb --full-freeze
```

Appends a complete `pip freeze` snapshot after the targeted manifest, for cases where you want a full bit-for-bit fallback available. Off by default — a full freeze is generally too much noise for a data scientist audience, and top-level pins are good guidance most of the time.

## Output

The tool prints two blocks to paste into new cells at the top of your notebook:

1. **Markdown cell** — explains the setup, and if applicable, notes hardware-tagged builds and GPU/accelerator usage detected during authoring.
2. **Code cell** — checks the Python version, writes a `pinned_requirements.txt`, and installs it via pip.

## Reading the GPU/accelerator messages

- **"Active accelerator detected"**: a GPU-relevant library was imported and a GPU/MPS/TPU was available and confirmed for that framework in your session. This is a strong signal the notebook needs one, not certainty that every operation used it.
- **"Framework imported, but no active accelerator found"**: you imported a GPU-capable library, but no accelerator was available in this test run. If you intended to require a GPU, this run doesn't confirm that — it just tells you this particular run happened on CPU.
- When multiple GPU frameworks are imported together, only the framework(s) actually confirmed to have an accelerator are reflected in the device name — the message doesn't imply every listed framework had GPU access.

## Known limitations

- **Dynamic imports** (e.g. `importlib.import_module("some_pkg")`) are not detected by AST scanning, since the module name isn't a literal in the source.
- **Transitive dependencies are not pinned**, only top-level imports. Sub-dependencies can still drift between installs.
- **GPU checks confirm availability, not actual usage** — a GPU can be available and imported without every tensor operation running on it.
- **No Kaggle Docker image tag detection** — no documented environment variable exposes this from inside a running kernel.
- **Path A vs Path B can diverge**: Path A reflects only what's saved to disk; Path B reflects live kernel history, which can be ahead of disk (unsaved edits) or behind it (deleted cells still in `In`). They are not interchangeable views of the same state.
- **No automatic detection of interpreter/kernel mismatch**: Path A correlates against whatever environment is running the script, not necessarily the one the notebook's kernel used. This is not checked or warned about beyond printing the interpreter path — verifying the match is the author's responsibility.
- Hardware-tagged builds (`+cu121`, etc.) are flagged, not auto-corrected — pip's default wheel selection when reinstalling an untagged version is not guaranteed to match the original hardware target.

## Roadmap

- Package as an installable library rather than a standalone script.
- A separate static-only scanner for notebooks you don't own (no live execution, no environment correlation — file-based AST scan only).
