#!/usr/bin/env python3
"""
Phase 5g: Live-Kernel Stale Module Test.
Generates a manifest using notebook_env.py, then simulates a user importing a 
package interactively, running the generated setup cell to re-pin it, and 
attempting to use it without restarting the kernel.
"""

import queue
import sys
import json
import subprocess
import os
import jupyter_client

def execute_and_capture(client, code, timeout=120):
    """Sends code to the kernel and captures the output from the iopub channel."""
    msg_id = client.execute(code)
    outputs, errors = [], []
    
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
                
            msg_type = msg['msg_type']
            content = msg['content']
            
            if msg_type == 'stream':
                outputs.append(content['text'])
            elif msg_type in ('execute_result', 'display_data'):
                if 'data' in content and 'text/plain' in content['data']:
                    outputs.append(content['data']['text/plain'])
            elif msg_type == 'error':
                errors.append(f"{content['ename']}: {content['evalue']}")
            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break
        except queue.Empty:
            errors.append("TimeoutError: Kernel execution timed out.")
            break
            
    return "".join(outputs), errors

def main() -> int:
    # 1. Generate the test fixture and use the tool to parse it
    fixture_path = "tests/fixtures/temp_stale_fixture.ipynb"
    
    # We embed an explicit pin so notebook_env generates a strict downgrade
    fixture_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["!pip install numpy==1.23.5\n", "import numpy"],
                "metadata": {}
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    os.makedirs("tests/fixtures", exist_ok=True)
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture_content, f)
        
    print("1. Running notebook_env.py to generate the setup cell...")
    try:
        # Use --in-place to completely avoid output-path resolution ambiguity
        result = subprocess.run(
            [sys.executable, "notebook_env.py", fixture_path, "--in-place"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"FAIL: notebook_env.py execution failed.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
        return 1

    with open(fixture_path, "r", encoding="utf-8") as f:
        out_nb = json.load(f)
        
    if len(out_nb.get("cells", [])) < 2:
        print(f"FAIL: notebook_env.py did not insert the setup cells.\nTool Output:\n{result.stdout}\n{result.stderr}")
        return 1

    # Cell 0 is markdown, Cell 1 is the sequential installer
    setup_cell_code = "".join(out_nb["cells"][1]["source"])
    
    # Clean up temp file
    os.remove(fixture_path)

    print("\nStarting interactive kernel...")
    km = jupyter_client.KernelManager(kernel_name='python3')
    km.start_kernel()
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=10)
    
    try:
        print("2. Importing numpy interactively...")
        out1, err1 = execute_and_capture(client, "import numpy\nprint(numpy.__version__)")
        if err1:
            print(f"FAIL: Setup failed to import numpy.\n{err1}")
            return 1
        initial_version = out1.strip()
        print(f"   Initial version loaded: {initial_version}")
        
        print("\n3. Executing generated Cell 2...")
        out2, err2 = execute_and_capture(client, setup_cell_code)
        if err2:
            print(f"FAIL: Kernel error during generated setup cell execution.\n{err2}")
            return 1
            
        advisory_text = "you may need to restart the kernel"
        if advisory_text not in out2.lower():
            print(f"FAIL: pip restart advisory was swallowed or missing.\nOutput:\n{out2}")
            return 1
        print("   PASS: pip kernel-restart advisory is correctly surfaced in output.")
        
        print("\n4. Checking active module state post-install...")
        out3, err3 = execute_and_capture(client, "import numpy\nprint(numpy.__version__)")
        if err3:
            print(f"FAIL: Kernel error during post-install check.\n{err3}")
            return 1
            
        stale_version = out3.strip()
        print(f"   Version currently active in memory: {stale_version}")
        
        if stale_version != initial_version:
            print(f"FAIL: Kernel somehow hot-reloaded the package. Expected {initial_version}, got {stale_version}")
            return 1
            
        print("   PASS: Module behaves as stale, confirming the live-session risk.")
        return 0
        
    finally:
        client.stop_channels()
        km.shutdown_kernel()

if __name__ == "__main__":
    sys.exit(main())