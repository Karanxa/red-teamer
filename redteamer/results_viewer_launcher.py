#!/usr/bin/env python
"""
Launcher script for the Red Teaming results viewer with Streamlit.

This script handles compatibility issues between PyTorch and Streamlit,
ensuring the application runs smoothly.
"""

import os
import sys
import subprocess
from pathlib import Path

# Make this script runnable from any location
script_dir = Path(__file__).parent
os.chdir(script_dir.parent)

def main():
    """Run the results viewer application."""
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

# Import the results viewer
from redteamer.results_viewer import main

# Run the results viewer
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())
    """
    
    # Write the temporary script
    temp_script_path = Path("temp_results_viewer_launcher.py")
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    try:
        # Run the script with streamlit
        print("Launching Red Teaming results viewer...")
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