
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
from redteamer.red_team.streamlit_runner import run_conversational_redteam_with_ui

# Parse arguments
class Args:
    def __init__(self):
        self.target_type = "ollama"
        self.model = "gemma3:1b"
        self.system_prompt = None
        self.chatbot_context = "You are a helpful AI assistant."
        self.redteam_model_id = 'first'
        self.max_iterations = 5
        self.verbose = True
        self.curl_command = None

# Run the conversational red teaming
if __name__ == "__main__":
    try:
        args = Args()
        run_conversational_redteam_with_ui(args)
    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())
    