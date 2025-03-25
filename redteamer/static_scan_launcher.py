#!/usr/bin/env python
"""
Launcher script for static model red teaming with PyTorch and Streamlit.

This script handles compatibility issues between PyTorch and Streamlit,
ensuring the application runs smoothly.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Make this script runnable from any location
script_dir = Path(__file__).parent
os.chdir(script_dir.parent)

def main():
    """Run the static scan red teaming application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Static LLM Red Teaming Scan")
    parser.add_argument("--provider", help="Model provider (openai, anthropic, gemini, ollama)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--custom-model", help="Custom model curl command with {prompt} placeholder")
    parser.add_argument("--custom-model-name", default="custom-model", help="Name for the custom model")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of adversarial prompts to generate")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set environment variables to fix PyTorch-Streamlit compatibility issues
    os.environ["REDTEAMER_FIX_STREAMLIT"] = "1"
    
    # Create a temporary script to launch Streamlit
    temp_script = """
import os
import sys
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

# Import the streamlit fix and apply it
from redteamer.utils.streamlit_fix import apply_torch_streamlit_fix
apply_torch_streamlit_fix()

# Import the streamlit runner
from redteamer.static_scan.streamlit_runner import run_static_scan_with_ui

# Parse arguments
class Args:
    def __init__(self):
        self.provider = {provider}
        self.model = {model}
        self.custom_model = {custom_model}
        self.custom_model_name = {custom_model_name}
        self.num_prompts = {num_prompts}
        self.output_dir = {output_dir}
        self.verbose = {verbose}

# Run the static scan
if __name__ == "__main__":
    try:
        args = Args()
        run_static_scan_with_ui(args)
    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Error: {{e}}")
        st.code(traceback.format_exc())
    """.format(
        provider=repr(args.provider),
        model=repr(args.model),
        custom_model=repr(args.custom_model),
        custom_model_name=repr(args.custom_model_name),
        num_prompts=args.num_prompts,
        output_dir=repr(args.output_dir),
        verbose=repr(args.verbose)
    )
    
    # Write the temporary script
    temp_script_path = Path("temp_static_scan_launcher.py")
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    try:
        # Run the script with streamlit
        print("Launching static scan red teaming tool...")
        subprocess.run(["streamlit", "run", str(temp_script_path)], check=True)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if temp_script_path.exists():
            temp_script_path.unlink()

if __name__ == "__main__":
    main() 