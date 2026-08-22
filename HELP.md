# Notebook Help

You're running a notebook, and something at the top of it, before your actual work starts, printed a message you don't understand. That's what this page is for.

You don't need to know anything about how that setup part of the notebook works. Find the message below (or use Ctrl+F / Cmd+F to search for a phrase from it) and read what to do about it.

---

## If the second cell (the one with code in it) printed something

This is usually where you'll end up. That cell installs some packages your notebook needs before the rest of it can run.

**"Error: Major Python version mismatch!"**

This notebook needs a different major version of Python than what you're currently running (for example, Python 3 vs. an old Python 2, which is now rare). It stops here on purpose, because the rest of the notebook usually won't work at all otherwise.

_What to do:_ you'll need to switch to a Python version that matches what the message says the notebook needs, then run the notebook again from the top. If you're not sure how to do that on whatever platform you're using (your own computer, Kaggle, Colab, etc.), that's worth a quick search for "how to change Python version" plus the platform's name.

**"This code was created with Python 3.X. You are trying to run it with 3.Y."**

A smaller mismatch, close enough that the notebook keeps going anyway. This is just a heads-up, not a stop.

_What to do:_ usually nothing. If something else fails further down in a way that doesn't match anything else on this page, come back and consider this the likely reason.

**"Note: If you see 'Retrying...' messages below while offline, enable Internet access and re-run this cell."**

Just what it says, installing packages needs internet access.

_What to do:_ if you keep seeing "Retrying" and nothing progresses, check your internet connection, then run that cell again.

**"Setup complete! Environment ready."**

Good news, everything installed correctly.

_What to do:_ nothing. Continue running the rest of the notebook.

**"Setup failed while installing pinned dependencies."**

Something didn't install correctly. If the message right after this one mentions a graphics-card-specific version mismatch, that's genuinely likely to be the cause, the tool only shows that specific suggestion when your notebook actually has a hardware-tagged package in its list. If it doesn't mention that, the real reason is something else.

_What to do:_ scroll up, above this failure message, to the actual error text from the install step. That's where the real reason is, usually one of: no internet connection, a package name that's changed or no longer exists, a package name that looks like a local file or folder rather than something published (worth asking whoever gave you this notebook, in that case), or a permissions/disk-space problem on your machine. If you genuinely can't tell what went wrong from the text above, the person who gave you this notebook is your best next step, they'll know what it's supposed to need.

---

## If the first cell (the description) mentions any of these

This first cell is just an explanation, it doesn't install anything itself, but it can mention a few things worth knowing before you hit run:

**"This notebook was created using an active accelerator..."**

The person who made this notebook had a graphics card (GPU) available, which speeds up heavy computations.

_What to do:_ if your computer doesn't have one, the notebook will most likely still run, just more slowly. Not a problem to fix, just something to expect.

**"Specific Package Builds Detected"**

A heads-up that one of the packages needs a version matched to specific hardware. If installation fails later, see "Setup failed" above.

**"Network Notice"**

A reminder that an internet connection is usually needed for the install step.

---

## If you opened the second cell and see lines starting with `#`

Those are notes, not instructions, anything starting with `#` is skipped, not installed. You don't need to do anything with these, but here's what they mean if you're curious:

| What it looks like                                  | What it means                                                                                                                                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `# package (... optional fallback)`                 | This package is only used as a backup option in the notebook, so it wasn't required.                                                                                                                                      |
| `# package (platform pseudo-module ...)`            | Not a real installable package, it's something automatically provided by the platform you're running on (like Kaggle or Databricks). Nothing to install.                                                                  |
| `# package (local repo module; not a PyPI package)` | This is a file that's meant to be included right alongside the notebook, not something from the internet. If something's missing, check that all the notebook's files were shared with you, not just the notebook itself. |
| `# tool (... not found in active env)`              | Something the notebook tries to use couldn't be confirmed as installed. Worth mentioning to whoever shared the notebook with you if it seems related to what's failing.                                                   |

---

## Still stuck?

This page covers the most common messages, not every possible way a package install can go wrong, those come from Python's own installer and can vary a lot. If nothing here matches what you're seeing:

1. Read the full error text, not just the summary line, the real reason is almost always printed just above it.
2. Ask whoever gave you this notebook. They know what it's supposed to need and are likely your fastest path to an answer.

---

_This page is a general reference for the setup cells this tool generates. If something you're seeing doesn't match what's described here, the notebook may be running an older or newer version of the tool than this page was written for — the person who shared the notebook with you is your best source for specifics._
