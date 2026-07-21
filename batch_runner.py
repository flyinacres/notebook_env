import argparse
import glob
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_single_notebook(notebook_path, compiler_code, full_freeze=False):
    """Executes a single notebook via nbconvert ExecutePreprocessor and appends the v15 compiler cell."""
    filename = os.path.basename(notebook_path)
    print(f"\n{'='*70}")
    print(f"⚡ TESTING: {filename}")
    print(f"{'='*70}")

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"❌ Failed to load notebook JSON: {e}")
        return False

    # Append compiler payload to the notebook AST
    execution_call = (
        f"\n\n# --- Injected Environment-Lock Call ---\n"
        f"generate_production_blueprint(full_freeze={full_freeze})\n"
    )
    full_payload = compiler_code + execution_call
    compiler_cell = nbformat.v4.new_code_cell(full_payload)
    nb.cells.append(compiler_cell)

    # Execute inside headless IPython kernel
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(
            nb, {"metadata": {"path": os.path.dirname(notebook_path) or "."}}
        )

        # Output capturing from compiler cell
        last_outputs = nb.cells[-1].outputs
        printed_anything = False
        for out in last_outputs:
            if out.output_type == "stream":
                print(out.text.strip())
                printed_anything = True
            elif out.output_type == "error":
                print(
                    f"❌ COMPILER EXECUTION ERROR: {out.ename} - {out.evalue}"
                )
                printed_anything = True

        if not printed_anything:
            print("⚠️ Compiler executed but produced no stdout.")

        return True

    except Exception as e:
        print(f"🚨 HEADLESS EXECUTION CRASH: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Project Environment-Lock: Batch Test Harness for Notebook Snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python batch_runner.py
  python batch_runner.py --full-freeze
  python batch_runner.py --file test_corpus/sample.ipynb
  python batch_runner.py --corpus-dir ./my_test_suite --compiler compiler_v15.py
""",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to a single .ipynb file to run instead of the full test corpus.",
    )
    parser.add_argument(
        "-c",
        "--compiler",
        type=str,
        default="compiler_v15.py",
        help="Path to the compiler script (default: compiler_v15.py).",
    )
    parser.add_argument(
        "-d",
        "--corpus-dir",
        type=str,
        default=os.path.join(".", "test_corpus"),
        help="Directory containing target test notebooks (default: ./test_corpus).",
    )
    parser.add_argument(
        "--full-freeze",
        action="store_true",
        help="Append a complete system snapshot (all installed packages) to generated manifests.",
    )

    args = parser.parse_args()

    # Load compiler file
    if not os.path.exists(args.compiler):
        print(
            f"❌ Error: Could not find compiler script at '{args.compiler}'."
        )
        sys.exit(1)

    with open(args.compiler, "r", encoding="utf-8") as f:
        compiler_code = f.read()

    # Single notebook target execution
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Error: Notebook '{args.file}' not found.")
            sys.exit(1)
        notebooks = [args.file]
    else:
        # Full corpus execution
        if not os.path.exists(args.corpus_dir):
            print(
                f"❌ Error: Test corpus directory '{args.corpus_dir}' does not exist."
            )
            sys.exit(1)

        notebooks = glob.glob(
            os.path.join(args.corpus_dir, "**", "*.ipynb"), recursive=True
        )

    if not notebooks:
        print(
            f"⚠️ No .ipynb files found in target location. Place test files there and re-run."
        )
        sys.exit(0)

    print(
        f"🚀 Starting batch evaluation across {len(notebooks)} notebook(s)..."
    )
    print(f"📌 Using compiler: {args.compiler}")
    print(f"📌 Full Freeze mode: {args.full_freeze}")

    passed = 0
    failed = 0

    for nb_path in notebooks:
        success = run_single_notebook(
            nb_path, compiler_code, full_freeze=args.full_freeze
        )
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {passed} PASSED | {failed} FAILED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()