#!/usr/bin/env python3
"""
Phase 0: Live-Kernel Regressions Test.
Simulates a user executing normal cells, followed by a paste-and-run of the 
entire notebook_env.py script into the same active kernel.
Verifies argv isolation, duplicate log handler prevention, and history filtering.
"""

import queue
import sys
import jupyter_client
import os

def execute_and_capture(client, code, timeout=30):
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
    tool_path = "notebook_env.py"
    if not os.path.exists(tool_path):
        print(f"FAIL: Cannot find {tool_path} in workspace.")
        return 1
        
    with open(tool_path, "r", encoding="utf-8") as f:
        tool_source = f.read()

    print("Starting interactive kernel...")
    km = jupyter_client.KernelManager(kernel_name='python3')
    km.start_kernel()
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=10)
    
    try:
        print("1. Populating kernel history with dummy imports...")
        # We use a 3rd party package so it survives the tool's stdlib filter
        out1, err1 = execute_and_capture(client, "import ipykernel\nprint('Dummy cell executed')")
        if err1:
            print(f"FAIL: Dummy execution failed.\n{err1}")
            return 1

        print("\n2. Executing notebook_env.py (Paste-and-Run 1)...")

        sentinel = "['notebook_env.py', '--test-sentinel']"
        set_sentinel_code = f"import sys\nsys.argv = {sentinel}"
        out_set, err_set = execute_and_capture(client, set_sentinel_code)
        if err_set:
            print(f"FAIL: Could not set sys.argv sentinel.\n{err_set}")
            return 1

        out2, err2 = execute_and_capture(client, tool_source)
        if err2:
            print(f"FAIL: tool execution crashed. (Possible sys.argv contamination)\n{err2}")
            return 1

        argv_check_code = "import sys\nprint(sys.argv)"
        out_argv, err_argv = execute_and_capture(client, argv_check_code)
        if err_argv:
            print(f"FAIL: Could not read back sys.argv.\n{err_argv}")
            return 1

        if out_argv.strip() != sentinel:
            print(f"FAIL: sys.argv was mutated. Expected {sentinel}, got {out_argv.strip()}")
            return 1

        print("   PASS: sys.argv contamination avoided (state verified unchanged).")

        print("\n3. Verifying History Introspection...")
        if "ipykernel" not in out2.lower():
            print("FAIL: Tool failed to extract prior cell history (ipykernel not found in output).")
            # Truncate out2 so it doesn't wash out the terminal
            print(f"--- Output Preview (last 1000 chars) ---\n{out2[-1000:]}\n----------------------------------------")
            return 1
            
        print("   PASS: History introspection correctly captured prior cells.")

        print("\n4. Executing notebook_env.py (Paste-and-Run 2 for Log Handlers)...")
        out3, err3 = execute_and_capture(client, tool_source)
        if err3:
            print(f"FAIL: Second execution crashed.\n{err3}")
            return 1

        print("\n5. Checking active log handlers...")
        handler_check = "import logging\nprint(len(logging.getLogger('notebook_env').handlers))"
        out4, err4 = execute_and_capture(client, handler_check)
        if err4:
            print(f"FAIL: Handler check failed.\n{err4}")
            return 1
            
        try:
            handler_count = int(out4.strip())
        except ValueError:
            print(f"FAIL: Could not parse handler count from output: {out4}")
            return 1

        if handler_count > 1:
            print(f"FAIL: Duplicate log handlers detected ({handler_count} handlers active).")
            return 1
            
        print("   PASS: Duplicate log handlers successfully prevented.")
        return 0
        
    finally:
        client.stop_channels()
        km.shutdown_kernel()

if __name__ == "__main__":
    sys.exit(main())