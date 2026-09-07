#!/usr/bin/env python3
"""
Phase 5g: Live-Kernel Stale Module Test.
Simulates a user importing a package interactively, running a generated setup cell
that re-pins it, and attempting to use it without restarting the kernel.
"""

import queue
import sys
import jupyter_client

def execute_and_capture(client, code, timeout=30):
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
    print("Starting interactive kernel...")
    km = jupyter_client.KernelManager(kernel_name='python3')
    km.start_kernel()
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=10)
    
    try:
        # Step 1: Pre-load the module in the active session
        print("1. Importing numpy interactively...")
        out1, err1 = execute_and_capture(client, "import numpy\nprint(numpy.__version__)")
        if err1:
            print(f"FAIL: Setup failed to import numpy. {err1}")
            return 1
        initial_version = out1.strip()
        print(f"   Initial version loaded: {initial_version}")
        
        # Step 2: Simulate executing the generated Cell 2 (re-pinning to an older version)
        print("\n2. Executing %pip install (simulating generated Cell 2)...")
        # We pin to an arbitrary older version that is guaranteed to be different
        out2, err2 = execute_and_capture(client, "%pip install numpy==1.23.5")
        
        # Verify the pip restart advisory is NOT swallowed
        advisory_text = "you may need to restart the kernel"
        if advisory_text not in out2.lower():
            print(f"FAIL: pip restart advisory was swallowed or missing.\nOutput:\n{out2}")
            return 1
        print("   PASS: pip kernel-restart advisory is correctly surfaced in output.")
        
        # Step 3: Verify the module is genuinely stale in the current session
        print("\n3. Checking active module state post-install...")
        out3, err3 = execute_and_capture(client, "import numpy\nprint(numpy.__version__)")
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