#!/usr/bin/env python3
import queue
import jupyter_client

def execute_and_capture(client, code, timeout=10):
    """Sends code to the kernel and captures the output from the iopub channel."""
    msg_id = client.execute(code)
    
    outputs = []
    errors = []
    
    while True:
        try:
            # Poll the iopub channel for cell output
            msg = client.get_iopub_msg(timeout=timeout)
            msg_type = msg['msg_type']
            content = msg['content']
            
            # Ignore broadcast messages not belonging to our execution
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
                
            if msg_type == 'stream':
                outputs.append(content['text'])
            elif msg_type in ('execute_result', 'display_data'):
                if 'data' in content and 'text/plain' in content['data']:
                    outputs.append(content['data']['text/plain'])
            elif msg_type == 'error':
                errors.append({
                    'ename': content['ename'],
                    'evalue': content['evalue']
                })
            elif msg_type == 'status' and content['execution_state'] == 'idle':
                # Kernel has finished executing the cell
                break
                
        except queue.Empty:
            print("Timeout waiting for kernel output.")
            break
            
    return "".join(outputs), errors

def main():
    print("Starting interactive kernel...")
    km = jupyter_client.KernelManager(kernel_name='python3')
    km.start_kernel()
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=10)
    
    try:
        # 1. Test basic execution and stdout capture
        out, err = execute_and_capture(client, "print('Hello from the interactive kernel!')")
        print(f"Stdout capture: {out.strip()}")
        
        # 2. Test state persistence (crucial for Phase 5g)
        execute_and_capture(client, "test_var = 'State persists!'")
        out, err = execute_and_capture(client, "print(test_var)")
        print(f"State capture: {out.strip()}")
        
        # 3. Test structural error capture (like check_negative_fixture.py)
        out, err = execute_and_capture(client, "import missing_fake_package")
        if err:
            print(f"Error capture: {err[0]['ename']}: {err[0]['evalue']}")
            
    finally:
        client.stop_channels()
        km.shutdown_kernel()
        print("Kernel shut down.")

if __name__ == "__main__":
    main()