# Notebook Setup Help (v35)

You've landed here because a notebook set up by `notebook_env.py` printed a message you want more context on, either while someone was *generating* the setup cells, or while you're *running* a notebook that already has them.

This file matches `notebook_env.py` version v35. If the notebook you're looking at shows a different version number in its setup cells, some of the wording below might not match exactly, the tool is still being actively developed. Message text can change between versions.

Find the message below, or use your browser's find-on-page (Ctrl+F / Cmd+F) to search for a phrase from it.

---

## Messages from the setup cell (while your notebook is installing packages)

This is the most common reason to be here: you ran a notebook, its second cell tried to install some packages, and something printed that you didn't expect.

**`❌ Error: Major Python version mismatch!`**

The notebook was built for a different major Python version than the one you're currently running (for example, Python 3 vs. Python 2, which is now rare). Execution stops here on purpose, because code usually doesn't run correctly across a major version change.

*What to do:* switch to an environment or kernel running the Python version the message names, then re-run the notebook from the top.

**`⚠️ This code was created with Python 3.X. You are trying to run it with 3.Y.`**

A smaller mismatch than above, same major version, different minor version (e.g. 3.10 vs. 3.11). This is a warning only; the notebook keeps running.

*What to do:* usually safe to ignore. If something fails later in a confusing way that doesn't match anything else on this page, this is worth revisiting first.

**`Applying pinned environment manifest [...]`** and **`💡 Note: If you see 'Retrying...' messages below while offline, enable Internet access and re-run this cell.`**

Normal, expected output. It's telling you it's about to install the exact package versions the original author recorded, and reminding you this step usually needs internet access.

*What to do:* nothing, unless installation is genuinely stuck on repeated "Retrying" messages, in which case check your internet connection.

**`✅ Setup complete! Environment ready.`**

Everything installed successfully.

*What to do:* nothing. Continue running the rest of the notebook normally.

**`❌ Setup failed while installing pinned dependencies.`**

The install step returned an error. The message that follows this one currently always suggests the problem is a hardware-specific build (something like a GPU version tag ending in `+cu121`), **even when that's not actually what went wrong.** This is a known issue with the tool itself, not something wrong with your setup.

*What to do:* scroll up, above this message, to the raw output from the install step. That text names the actual package and reason it failed. Common real causes, roughly in order of likelihood: no internet connection, a package that no longer exists or was renamed, a disk space or permissions problem on your machine, and, yes, sometimes actually a hardware-build mismatch. If it does turn out to be hardware-related, try installing the plain version of that one package yourself in a new cell, e.g. `!pip install torch` (without a version tag), and see if that resolves it.

---

## Messages inside the package list itself

If you open the generated code cell or the `pinned_requirements.txt` file it creates, you'll see some lines starting with `#`. A `#` means that line is a comment, informational only, not something being installed.

| What it looks like | What it means |
|---|---|
| `# package (imported as 'x' in try/except or conditional block - optional fallback)` | The notebook only uses this package inside a fallback block (code that tries one approach and falls back to another if the first isn't available). Since it's optional by the original author's own design, it's listed but not force-installed. Nothing to fix. |
| `# package (platform pseudo-module provided by runtime environment)` | Not a real installable package, it's something the platform (Kaggle, Databricks, etc.) provides automatically while your notebook runs there. Can't be installed elsewhere and doesn't need to be. |
| `# package (local repo module; not a PyPI package)` | This is actually a folder of code sitting next to the notebook, not something from the internet. If it's missing, you're missing a file, not a package, check that it was included when the notebook was shared with you. |
| `# tool (installed via cell magic; not found in active env)` | Somewhere in the notebook, an install command ran for something that isn't a normal Python import. The tool couldn't confirm whether it's actually installed. Worth checking manually if the notebook seems to depend on it. |

---

## Messages in the explanation cell (the Markdown cell above the code)

**"Hardware Acceleration: This notebook was created using an active accelerator..."**

The person who made this notebook had a working GPU (specialized hardware for heavy computation) when they ran it.

*What to do:* if your machine doesn't have a GPU, the notebook may still run, just more slowly. Not an error.

**"Specific Package Builds Detected"**

At least one package needs a specific hardware-matched version. See the "Setup failed" entry above if this causes an install failure.

**"Network Notice"**

A reminder that installing packages usually needs internet access. Only relevant if the install step actually fails while you're offline.

---

## Messages you might see while generating a notebook's setup cells

These come from running `notebook_env.py` itself, if you're the one setting up (or re-setting-up) a notebook's dependencies, rather than just running a finished one.

| Message | What it means | What to do |
|---|---|---|
| `⚠️ HARDWARE BUILD WARNINGS: Specific hardware build detected` | One of your imported packages is pinned to a hardware-specific build with no known download source recorded. | If installs fail on another machine later, this is the first thing to check. You can also supply `--extra-index-url` when running the tool if you know the correct source. |
| `⚡ Active accelerator detected: [device name]` | A GPU-related library was imported, and a GPU was confirmed working when you ran the tool. | Informational, nothing to do. |
| `⚠️ Acceleration Framework (...) imported, but NO active accelerator detected` | A GPU-related library was imported, but no GPU was available in this specific run. | If the notebook is meant to require a GPU, this just means it wasn't confirmed this time, it doesn't necessarily mean the notebook is broken. |
| `💡 Extra Dependency Promotion: importing 'x.y' automatically promoted requirement to 'package[y]==...'` | Some packages have optional add-on features (called extras) that only install if specifically requested. The tool noticed your code needs one and included it automatically. | Informational, nothing to do. |
| `⚠️ Dynamic import detected via non-literal argument...; statically unresolvable` | Somewhere in the code, a package is loaded by a variable name rather than written directly, so the tool can't figure out in advance what it will be. | If you know what that variable resolves to, add that package to the generated list by hand. |
| `ℹ️ System package manager call detected` / `ℹ️ Conda installation detected` | The notebook installs something outside Python's normal package system (`apt-get`, `conda install`, etc.), which this tool doesn't track. | Note these separately if you're documenting the notebook's full requirements. |
| `⚠️ ...references an external requirements file; contents cannot be verified statically` | The notebook installs from a `requirements.txt`-style file the tool can't see inside. | Make sure that file is included and shared alongside the notebook. |

---

## Still stuck?

This page covers the messages the tool is known to produce, it doesn't cover every possible way a package install can fail in general (those failures come from `pip` and Python itself, not from this tool). If what you're seeing isn't listed above:

1. Read the raw error text in full, above whatever summary message pointed you here. The actual reason is almost always in that raw text.
2. If you're comfortable with more technical detail, `DEVELOPMENT.md` in this repository tracks known bugs and in-progress work, your issue may already be a known one.
3. If neither helps, the person who originally shared this notebook with you is likely your fastest path to an answer, they know what the notebook is actually supposed to do.

---

*This file covers `notebook_env.py` v35. See `README.md` for how the tool works more generally, and `DEVELOPMENT.md` for technical/roadmap detail.*
