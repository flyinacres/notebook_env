# Notebook Environment-Lock Tool (v18)

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

**Path A — against a saved `.ipynb` file (recommended):**

```bash
python env_lock.py your_notebook.ipynb
```

**Path B — pasted into a live notebook cell:**

Paste the tool's code into a cell and call `main()`, or adapt it to call `extract_from_active_session()` directly. This reads IPython's `In` history rather than the saved file, so it reflects whatever ran in the current kernel session — restart the kernel before this final run if you fixed anything earlier, or stale/removed cells will still be counted.

Do **not** invoke this tool via `import your_module; your_module.main()` inside the notebook you're scanning — the import line itself becomes part of the session history and may show up as a spurious entry in the generated manifest. Paste the tool inline instead.

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
- **Path A vs Path B can diverge**: Path A reads the saved `.ipynb` file from disk; Path B reads live kernel history. If you test interactively without saving intermediate states, these can disagree about what was actually exercised.
- Hardware-tagged builds (`+cu121`, etc.) are flagged, not auto-corrected — pip's default wheel selection when reinstalling an untagged version is not guaranteed to match the original hardware target.

## Roadmap

- Package as an installable library rather than a standalone script.
- A separate static-only scanner for notebooks you don't own (no live execution, no environment correlation — file-based AST scan only).
