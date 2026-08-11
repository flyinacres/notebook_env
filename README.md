# Notebook Environment Setup Tool

**What this does, in one sentence:** it looks at what your notebook actually needs to run, and gives you two cells to paste in so it still works the same way when someone else opens it.

You don't need to be a software engineer to use it. This guide assumes you know how to run a notebook cell and use `pip install` — nothing more. Anywhere this guide uses a term you might not know, it's explained the first time it comes up, and there's a short glossary at the end.

---

## Quick start

1. Finish your notebook. Run it from the top, in a fresh kernel (a fresh kernel means: restart it first, so nothing left over from earlier experimenting is still sitting in memory), so what you're about to check reflects what the notebook actually needs, not leftover state from something you tried and abandoned earlier.
2. Get the tool next to your work:
   - **On your own computer:** download `notebook_env.py` and put it in the same folder as your notebook.
   - **On Kaggle or Colab:** copy the entire contents of `notebook_env.py` and paste it into a new, empty cell in your notebook. It's a lot of code, but you will be able to delete it after running it.
3. Run it:
   - **On your own computer**, open a terminal in that folder and run:
     ```bash
     python notebook_env.py your_notebook.ipynb
     ```
   - **On Kaggle or Colab**, just run the cell you pasted it into.
4. It prints two blocks of text. Copy the first block into a new **Markdown** cell, and the second block into a new **Code** cell. Put both at the very top of your notebook, above everything else you've written. Now, if you pasted the notebook_env.py code into a cell, you can safely delete that cell. 
5. Save and share the notebook. When someone else opens it and runs it top to bottom, your new first two cells run first and set their environment up to match yours, automatically.

That's the whole workflow for the common case. Everything past this point is either "here's what those two cells actually contain and mean," or "here's what to do if something looks off" — you can stop reading here and come back later if you hit a question.

---

## What you'll get back

Here's a real (shortened) example of the two cells it hands you.

**Cell 1 (Markdown) — explains what's about to happen:**

> ### 🛠️ Environment Setup & Dependency Verification
>
> This notebook includes a pinned environment manifest (`pinned_requirements.txt`) to ensure reproducible execution.
>
> - **Dependency Sync:** Cell 2 will verify your active Python version and apply the exact package manifest recorded by the author.
> - **Hardware Acceleration:** This notebook was created using an active accelerator (`NVIDIA GeForce RTX 3090`, verified via PyTorch). _(only appears if a GPU was actually detected — see "Messages you might see" below)_

**Cell 2 (Code) — actually checks and installs things:**

```python
# =====================================================================
# VERIFIED ENVIRONMENT DEPENDENCIES (2026-08-05 12:00:00)
# =====================================================================
import sys, subprocess

REQUIRED_PYTHON = (3, 11)
# ... checks your Python version matches, warns (but doesn't stop) on a small mismatch ...

requirements_content = """# Tested top-level packages for this notebook
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
# cupy (imported as 'cupy' in try/except or conditional block - optional fallback)
"""
# ... writes that list to a file called pinned_requirements.txt, then installs it with pip ...
```

Two things worth pointing out in that example, since they show up a lot:

- `==1.26.4` means "exactly this version, no substitutes." This is called **pinning** — it's how the tool guarantees the same versions the author actually tested with, instead of "whatever's newest today."
- The `cupy` line is commented out (it starts with `#`, so it's ignored rather than installed). That's not a mistake. It means the original notebook only used `cupy` inside a fallback block (a `try/except`, explained below) — so it's optional, not something every reader is forced to install.

---

## Messages you might see, and what to do about them

This is the reference section. You won't see all of these on any single run — skim for whatever you're actually looking at.

### In the explainer cell (Cell 1)

| What it says                                                             | What it means                                                                                                                                                      | What to do                                                                                                                                    |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hardware Acceleration:** ...verified via PyTorch (or TensorFlow / JAX) | The notebook was authored on a machine with a GPU (a graphics card used to speed up heavy computation), and that GPU was actually confirmed working at the time.   | If you're running on a machine without a GPU, the notebook may run correctly but much more slowly. This isn't an error — it's advance notice. |
| **Specific Package Builds Detected**                                     | One of your packages (usually `torch`, `tensorflow`, or similar) was built for a specific hardware setup, e.g. a version number ending in something like `+cu121`. | If installation fails later, this is the first thing to suspect — see "Setup failed" below.                                                   |
| **Network Notice**                                                       | A reminder that installing packages usually needs an internet connection, unless they're already cached in your environment.                                       | Nothing to do unless the install step actually fails while offline.                                                                           |

### While Cell 2 runs

| What it says                                                                     | What it means                                                                                                                                                                                                                                                                                                                                                                                                     | What to do                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **❌ Error: Major Python version mismatch**                                      | The notebook was written for a different major Python version (e.g. Python 3 vs. Python 2 — a very rare mismatch today) than the one you're running it with. This stops execution, because it usually won't work at all.                                                                                                                                                                                          | Switch to a kernel/environment running the required Python version.                                                                                                                                                                                                                                                                                                                     |
| **⚠️ This code was created with Python 3.X. You are trying to run it with 3.Y.** | A smaller mismatch — same major version, different minor version (e.g. 3.10 vs. 3.11). This is a warning, not a stop; execution continues.                                                                                                                                                                                                                                                                        | Usually fine to ignore and just try running it. If something breaks in a confusing way later, this is worth revisiting.                                                                                                                                                                                                                                                                 |
| **✅ Setup complete! Environment ready.**                                        | Everything installed successfully.                                                                                                                                                                                                                                                                                                                                                                                | Nothing — you're done, continue running your notebook normally.                                                                                                                                                                                                                                                                                                                         |
| **❌ Setup failed while installing pinned dependencies.**                        | The `pip install` step returned an error. **Known caveat as of this version:** the tool always suggests the failure is about hardware-specific builds (like GPU version tags), even when it isn't. If your notebook doesn't use any GPU-specific packages, that particular suggestion doesn't apply to you — the real reason is in the pip output printed just above this message; scroll up and read that first. | Look at the pip error text right above this message for the actual reason (a typo'd package name, no internet connection, a disk-space or permissions issue, and so on, are all more common in practice than a hardware mismatch). If it does turn out to be hardware-related, try installing the plain (non-hardware-tagged) version instead, e.g. `!pip install torch` in a new cell. |

### Inside the generated package list itself

These show up as comments (lines starting with `#`) inside the list of packages the tool writes out. A comment means "informational, not installed automatically."

| What it looks like                                                                   | What it means                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `# package (imported as 'x' in try/except or conditional block - optional fallback)` | The notebook only imports this package inside a fallback block — code written to try one approach and fall back to another if the first isn't available. Since the notebook doesn't strictly require it, it's listed but not force-installed. |
| `# package (platform pseudo-module provided by runtime environment)`                 | This isn't a real installable package — it's something Kaggle, Databricks, or a similar platform automatically provides while your notebook is running there. Installing it yourself isn't possible and isn't needed.                         |
| `# package (local repo module; not a PyPI package)`                                  | This "package" is actually a folder of code that lives right next to the notebook, not something from the internet. Nothing to install — it just needs to be copied along with the notebook.                                                  |
| `# tool (installed via cell magic; not found in active env)`                         | Something in the notebook ran an install command (like `%pip install`) for a tool that isn't a regular Python import — the tool couldn't confirm it's currently installed. Worth double-checking manually if the notebook depends on it.      |

### GPU / accelerator messages specifically

| What it says                                                               | What it means                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **⚡ Active accelerator detected: [device name]**                          | A GPU-related library was imported, and a GPU was confirmed available and working when the author ran it. Strong signal the notebook benefits from a GPU — not a guarantee every single step used it.                          |
| **⚠️ Acceleration Framework imported, but NO active accelerator detected** | A GPU-related library was imported, but no GPU was actually available in that run. If the notebook was meant to require a GPU, this particular run just didn't confirm that — it doesn't necessarily mean it doesn't need one. |

**Known caveat:** if a notebook uses more than one GPU library at once (for example both `torch` and `tensorflow`), the device name shown can currently get attributed to the wrong one of the two. Treat the specific device name as a helpful hint rather than something to fully rely on for now.

**Also worth knowing:** if your notebook uses a higher-level library that relies on `torch`/`tensorflow`/`jax` underneath (for example `fastai`, which is built on `torch`), GPU detection may currently miss it even though a GPU actually was used. This is a known gap, not something you did wrong.

### Extra messages

| What it says                                                                                               | What it means                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **💡 Extra Dependency Promotion:** importing 'x.y' automatically promoted requirement to 'package[y]==...' | Some packages have optional add-on features (called **extras**) that only get installed if you ask for them specifically. The tool noticed your notebook uses one of those add-ons and automatically included it, rather than leaving it out. Informational only.                                                                                                    |
| **⚠️ Dynamic import detected via non-literal argument...; statically unresolvable**                        | Somewhere in the notebook, code loads a package by name using a variable instead of writing the package name directly (e.g. loading whatever package name is stored in a variable, decided while the code runs, rather than typing `import pandas` directly). The tool can't figure out in advance what that will be, so it can't check or include it automatically. | If you know what that variable resolves to, add that package to the list by hand.                                                                |
| **ℹ️ System package manager call detected** / **ℹ️ Conda installation detected**                           | The notebook installs something outside of Python's normal package system (e.g. `apt-get`, or a `conda install` command). These are outside what this tool tracks.                                                                                                                                                                                                   | Make a note of these separately if you're documenting the notebook's full requirements — they won't appear in the generated Python package list. |
| **⚠️ ...references an external requirements file; contents cannot be verified statically**                 | The notebook installs from a `requirements.txt` or similar file rather than listing packages directly. The tool can't see inside that file.                                                                                                                                                                                                                          | Check that file is included and shared alongside the notebook.                                                                                   |

---

## Two ways to run it

There are two ways to use this tool, and picking the right one matters.

**File mode** _(called "Path A" in some places, including the technical notes)_ — you run it from a terminal against your saved `.ipynb` file, like in the Quick Start above. It looks at exactly what's saved in that file. If you made changes and haven't saved yet, it won't see them.

**Paste mode** _(called "Path B")_ — you paste the tool directly into a notebook cell and run it there. This looks at everything that has actually run in your current session, whether or not you've saved. This is usually the better fit on Kaggle, Colab, or any cloud notebook where getting to a terminal is more hassle than just running one more cell.

**A trap to know about in Paste mode:** it looks at _everything you've run in this session_, including things you tried and later abandoned. If you experimented with a package earlier, decided against it, and never restarted your kernel, that abandoned attempt can still show up in the results. **Fix:** restart your kernel and run all cells fresh, right before you run this tool, so only your actual final code counts.

**A caution for File mode on your own computer:** if you use multiple Python environments (for example different projects with different setups), make sure you're running the tool from the same environment your notebook actually uses. The tool has no way to detect if you're in the wrong one — it will still produce a normal-looking result, just potentially the wrong one. It prints the Python location it's using at the top of its output specifically so you can double check this.

---

## Scanning many notebooks at once (for instructors)

If you're checking an entire course folder or shared repository rather than one notebook, point the tool at the folder instead of a single file:

```bash
python notebook_env.py --batch ./course_materials
```

This gives you a summary report across every notebook in that folder — what's missing, what's inconsistent, and which notebooks have problems — without changing any files. It's meant for auditing, not for the single-notebook "paste two cells in" workflow above.

There are a few more options for this mode (writing a combined package list for the whole folder, or actually inserting the setup cells into every notebook automatically). See `DEVELOPMENT.md` if you need those — they're less commonly needed and more worth understanding in detail before using.

**Known gap:** this folder-wide summary currently can't flag the hardware-specific build tag issue described above (the `+cu121`-style version warning) the way the single-notebook workflow can. If a notebook in your folder has that issue, batch mode won't currently tell you.

---

## Things it can't do yet

Being upfront about this rather than letting you discover it the hard way:

- If a package is loaded by name from a variable (rather than written directly, e.g. `import pandas`), and that variable's value isn't obvious from reading the code, the tool can't figure out what it is.
- A specific, less common style of import (`from . import something`) isn't detected at all.
- It only checks packages you import directly — not the packages _those_ packages depend on internally. Those can still change version on their own between installs.
- It confirms a GPU was _available_, not that every part of the notebook actually used it.
- It can't automatically fix a hardware-specific build mismatch (like a GPU version tag) — only flag it.
- Running it against a saved file vs. pasting it into a live notebook can genuinely give different answers (see "Two ways to run it" above) — they're not interchangeable, and picking the wrong one for your situation can produce a misleading result.

For the full technical list, ongoing work, and known internal bugs being tracked, see `DEVELOPMENT.md` — that document is written for whoever's actively developing this tool, not for day-to-day users, so it's denser than this guide on purpose.

---

## A few terms explained

- **Dependency / package:** a piece of code someone else wrote that your notebook uses, installed with `pip install`.
- **Pinning:** locking a package to one exact version (`==1.26.4`) instead of "whatever's newest," so results stay reproducible over time.
- **Kernel:** the running Python process behind your notebook. "Restarting the kernel" clears everything in memory and starts fresh.
- **Environment:** the specific combination of Python version and installed packages you're currently working in.
- **Guarded / conditional import:** code that tries to import something, and has a planned fallback if that import fails — usually written as `try: import x except: ...`. Since it's optional by design, this tool treats it as optional too.
- **Extras:** optional add-on features for a package that only install if specifically requested (e.g. `package[extra_name]`).
- **GPU / accelerator:** specialized hardware that speeds up heavy numerical work, commonly used for machine learning. "Accelerator" also covers Apple Silicon GPUs (MPS) and Google TPUs, not just NVIDIA GPUs.
- **Manifest:** the list of packages and versions this tool generates for your notebook.

---

_For the technical roadmap, in-progress work, and known internal bugs, see `DEVELOPMENT.md`._
