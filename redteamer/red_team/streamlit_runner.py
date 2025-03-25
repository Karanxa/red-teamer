"""
Standalone script to launch Streamlit UI for the Conversational Red-Teaming Scanner.

Run this script using:
    streamlit run redteamer/red_team/streamlit_runner.py -- [options]
"""

import os
import sys
import argparse
import time
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our Streamlit patch before importing streamlit
from redteamer.red_team.streamlit_patch import safe_import_streamlit

# Import streamlit safely
st = safe_import_streamlit()

# Import necessary modules
from redteamer.red_team.conversational_redteam import ConversationalRedTeam
from redteamer.red_team.streamlit_ui import (
    initialize_streamlit_app,
    display_config,
    display_model_loading_status,
    display_conversation,
    display_vulnerabilities,
    display_metrics,
    display_summary
)

def run_streamlit_app(args):
    """
    Launch the Streamlit app with the provided configuration.
    
    Args:
        args: Configuration object containing all necessary parameters
    """
    # Create a temporary script to run streamlit with the provided args
    script_content = f"""
import streamlit as st
import asyncio
import sys
import traceback
from redteamer.red_team.streamlit_runner import run_conversational_redteam_with_ui

def robust_runner():
    try:
        class Args:
            def __init__(self):
                self.target_type = "{args.target_type}"
                self.model = {repr(args.model)}
                self.curl_command = {repr(args.curl_command)}
                self.system_prompt = {repr(args.system_prompt)}
                self.chatbot_context = "{args.chatbot_context}"
                self.redteam_model_id = "{args.redteam_model_id}"
                self.hf_api_key = {repr(args.hf_api_key)}
                self.max_iterations = {args.max_iterations}
                self.output_dir = "{args.output_dir}"
                self.verbose = {args.verbose}
                self.quant_mode = "{args.quant_mode}"
        
        args = Args()
        asyncio.run(run_conversational_redteam_with_ui(args))
    except Exception as e:
        st.error(f"Critical error in Red Teaming process: {{str(e)}}")
        st.error("Please check the error details below:")
        st.code(traceback.format_exc())
        st.warning("Recommendation: Try running with --quant-mode cpu or use a smaller model.")

if __name__ == "__main__":
    robust_runner()
"""
    
    # Write the temporary script
    temp_script_path = "temp_streamlit_runner.py"
    with open(temp_script_path, "w") as f:
        f.write(script_content)
    
    try:
        # Launch Streamlit with the temporary script
        subprocess.run(["streamlit", "run", temp_script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
    finally:
        # Clean up the temporary script
        try:
            os.remove(temp_script_path)
        except:
            pass

async def run_conversational_redteam_with_ui(args):
    """
    Run the conversational red-teaming process with Streamlit UI.
    
    Args:
        args: Command line arguments
    """
    # Initialize the app
    initialize_streamlit_app()
    
    # Set up model configuration
    model_config = {}
    
    if args.target_type == "curl":
        model_config = {
            "curl_command": args.curl_command,
            "system_prompt": args.system_prompt
        }
    elif args.target_type == "ollama":
        model_config = {
            "model": args.model,
            "system_prompt": args.system_prompt,
            "temperature": 0.7,
            "max_tokens": 1000
        }
    else:
        model_config = {
            "provider": args.target_type,
            "model": args.model,
            "system_prompt": args.system_prompt,
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    # Display initial configuration
    display_config(
        target_model_type=args.target_type,
        chatbot_context=args.chatbot_context,
        redteam_model_id=args.redteam_model_id,
        max_iterations=args.max_iterations
    )
    
    # Initialize results dictionary
    results = {
        "conversation": [],
        "vulnerabilities": [],
        "start_time": time.time()
    }
    
    # Create shared data container
    if 'results' not in st.session_state:
        st.session_state.results = results
    
    # Create the conversational red-teaming engine
    display_model_loading_status("Loading the red-teaming model...")
    
    redteam = ConversationalRedTeam(
        target_model_type=args.target_type,
        chatbot_context=args.chatbot_context,
        redteam_model_id=args.redteam_model_id,
        hf_api_key=args.hf_api_key,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        quant_mode=args.quant_mode
    )
    
    try:
        # Load the model
        display_model_loading_status("Loading red-teaming model...")
        redteam._load_local_model()
        
        # Update configuration display to show fallback status if needed
        display_config(
            target_model_type=args.target_type,
            chatbot_context=args.chatbot_context,
            redteam_model_id=redteam.redteam_model_id,  # Use actual model ID which may have changed
            max_iterations=args.max_iterations,
            using_fallback=redteam.using_fallback,
            using_templates=hasattr(redteam, 'using_templates') and redteam.using_templates
        )
        
        if hasattr(redteam, 'using_templates') and redteam.using_templates:
            display_model_loading_status("Using template-based generation (no model loaded)")
        elif redteam.using_fallback:
            display_model_loading_status(f"Using fallback model {redteam.redteam_model_id}")
        else:
            display_model_loading_status("Model loaded successfully")
        
        # Configure the target model
        display_model_loading_status("Configuring target model...")
        redteam._configure_target_model(model_config)
        display_model_loading_status("Target model configured")
        
        # Run the red-teaming process
        display_model_loading_status("Starting conversational red-teaming process...")
        
        # Set up initial UI elements
        st.session_state.progress = st.progress(0)
        st.session_state.conversation_container = st.container()
        st.session_state.vuln_container = st.container()
        st.session_state.metrics_container = st.container()
        st.session_state.summary_container = st.container()
        
        # Start the conversation loop
        results = await redteam.run_redteam_conversation(model_config)
        
        # Update final results
        st.session_state.results = results
        
        # Display final summary
        display_summary(results)
        
    except Exception as e:
        st.error(f"Error during red-teaming process: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conversational Red-Teaming Scanner with Streamlit UI")
    parser.add_argument("--target-type", required=True, help="Target model type")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--curl-command", help="Curl command template")
    parser.add_argument("--system-prompt", help="System prompt")
    parser.add_argument("--chatbot-context", required=True, help="Chatbot context")
    parser.add_argument("--redteam-model-id", required=True, help="Red-teaming model ID")
    parser.add_argument("--hf-api-key", help="HuggingFace API key")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--output-dir", default="results/conversational", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quant-mode", default="auto", help="Quantization mode")
    
    args = parser.parse_args()
    asyncio.run(run_conversational_redteam_with_ui(args)) 