import glob
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_single_notebook(notebook_path, compiler_code):
    filename = os.path.basename(notebook_path)
    print(f"\\n{'='*70}")
    print(f"?? TESTING: {filename}")
    print(f"{'='*70}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"? Failed to load notebook JSON: {e}")
        return False

    # Dynamically append compiler script into execution sequence
    compiler_cell = nbformat.v4.new_code_cell(compiler_code)
    nb.cells.append(compiler_cell)

    # Execute inside headless IPython kernel
    ep = ExecutePreprocessor(timeout=300, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path) or '.'}})
        
        # Parse stdout stream from compiler cell
        last_outputs = nb.cells[-1].outputs
        printed_anything = False
        for out in last_outputs:
            if out.output_type == 'stream':
                print(out.text.strip())
                printed_anything = True
            elif out.output_type == 'error':
                print(f"?? COMPILER EXECUTION ERROR: {out.ename} - {out.evalue}")
                printed_anything = True
                
        if not printed_anything:
            print("?? Compiler executed but produced no stdout.")
            
        return True

    except Exception as e:
        print(f"?? HEADLESS EXECUTION CRASH: {e}")
        return False

def main():
    compiler_file = "compiler_v14.py"
    if not os.path.exists(compiler_file):
        print(f"? Error: Could not find '{compiler_file}' in working directory.")
        sys.exit(1)

    with open(compiler_file, "r", encoding="utf-8") as f:
        compiler_code = f.read()

    corpus_dir = os.path.join(".", "test_corpus")
    notebooks = glob.glob(os.path.join(corpus_dir, "**", "*.ipynb"), recursive=True)

    if not notebooks:
        print(f"?? No .ipynb files found in '{corpus_dir}'. Place test files there and re-run.")
        sys.exit(0)

    print(f"?? Starting batch evaluation across {len(notebooks)} notebook(s)...\\n")
    
    passed = 0
    failed = 0

    for nb_path in notebooks:
        success = run_single_notebook(nb_path, compiler_code)
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\\n{'='*70}")
    print(f"?? SUMMARY: {passed} PASSED | {failed} FAILED")
    print(f"{'='*70}\\n")

if __name__ == "__main__":
    main()
