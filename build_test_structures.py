"""
Generates structured test fixtures for local-module discovery, duplicate-stem
collision, and relative-asset mirroring tests.

Run from D:\\Dev\\notebook_env:
    python build_test_structures.py

Creates everything under .\\test_structures\\
"""
import json
import os

ROOT = "test_structures"

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
}


def make_notebook(path: str, code_lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": code_lines,
            }
        ],
        "metadata": KERNEL_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"  wrote {path}")


def make_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {path}")


print("Case 1: Subdirectory helpers (two variants: package-style vs sys.path hack)")
make_file(
    f"{ROOT}/case1_subdir_helper/project/code/helper.py",
    "def do_thing():\n    return 42\n",
)
make_notebook(
    f"{ROOT}/case1_subdir_helper/project/train_package_style.ipynb",
    ["from code import helper\n", "helper.do_thing()\n"],
)
make_notebook(
    f"{ROOT}/case1_subdir_helper/project/train_syspath_style.ipynb",
    ["import sys\n", "sys.path.append('code')\n", "import helper\n", "helper.do_thing()\n"],
)

print("\nCase 2: Parent/root helpers (confirms existing root_dir wiring)")
make_file(
    f"{ROOT}/case2_root_helper/repo/common.py",
    "def shared():\n    return 1\n",
)
make_notebook(
    f"{ROOT}/case2_root_helper/repo/notebooks/run.ipynb",
    ["import common\n", "common.shared()\n"],
)

print("\nCase 3: Duplicate stems across sibling directories")
make_notebook(
    f"{ROOT}/case3_duplicate_stems/dir_a/pipeline.ipynb",
    ["import pandas as pd\n", "# dir_a variant\n"],
)
make_notebook(
    f"{ROOT}/case3_duplicate_stems/dir_b/pipeline.ipynb",
    ["import numpy as np\n", "# dir_b variant\n"],
)

print("\nCase 4: Relative asset dependency")
make_file(
    f"{ROOT}/case4_relative_assets/analysis/data/data.csv",
    "a,b\n1,2\n",
)
make_notebook(
    f"{ROOT}/case4_relative_assets/analysis/run.ipynb",
    ["import pandas as pd\n", "df = pd.read_csv('data/data.csv')\n"],
)

print("\nDone. Structure:")
for dirpath, dirnames, filenames in os.walk(ROOT):
    depth = dirpath.replace(ROOT, "").count(os.sep)
    indent = "  " * depth
    print(f"{indent}{os.path.basename(dirpath) or ROOT}/")
    for fn in filenames:
        print(f"{indent}  {fn}")