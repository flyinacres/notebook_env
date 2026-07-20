# ?? Notebook Environment Lock: Batch Testing Harness

A headless, CLI-driven testing framework for running `generate_production_blueprint()` across large corpora of Jupyter notebooks (`.ipynb`).

---

## ?? Quick Start (Windows CMD Setup)

Follow these steps from standard Command Prompt (`cmd.exe`) after cloning this repository:

### 1. Set Up the Python 3.13 Virtual Environment
```cmd
:: Create virtual environment using Python 3.13
py -3.13 -m venv notebook_env

:: Activate the environment
notebook_env\Scripts\activate.bat

:: Upgrade pip and install execution harness dependencies
python -m pip install --upgrade pip
pip install ipykernel nbconvert papermill packaging# notebook_env
Determining necessary environment for a notebook
