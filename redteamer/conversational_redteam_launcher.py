#!/usr/bin/env python
"""
Launcher script for conversational red teaming with PyTorch and Streamlit.

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
    """Run the conversational red teaming application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational LLM Red Teaming")
    parser.add_argument("--target-type", default="ollama", help="Target model type (ollama, openai, anthropic, etc.)")
    parser.add_argument("--model", default="llama3", help="Model name")
    parser.add_argument("--system-prompt", help="System prompt for the target model")
    parser.add_argument("--chatbot-context", default="You are a helpful AI assistant.", help="Description of the chatbot being tested")
    parser.add_argument("--redteam-model-id", help="Model ID for red teaming (if not provided, will use default)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of conversation iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--curl-command", help="Custom curl command for target model")
    
    args = parser.parse_args()
    
    # Set environment variables to fix PyTorch-Streamlit compatibility issues
    os.environ["REDTEAMER_FIX_STREAMLIT"] = "1"
    
    # Create a temporary script to launch Streamlit
    temp_script = """
import os
import sys
import asyncio
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
        self.target_type = "{target_type}"
        self.model = "{model}"
        self.system_prompt = {system_prompt}
        self.chatbot_context = "{chatbot_context}"
        self.redteam_model_id = {redteam_model_id}
        self.max_iterations = {max_iterations}
        self.verbose = {verbose}
        self.curl_command = {curl_command}
        self.hf_api_key = None
        self.output_dir = "results/conversational"
        self.quant_mode = "auto"

# Run the conversational red teaming
if __name__ == "__main__":
    try:
        args = Args()
        # Properly await the coroutine using asyncio.run
        asyncio.run(run_conversational_redteam_with_ui(args))
    except Exception as e:
        import traceback
        import streamlit as st
        
        st.error(f"Error: {{e}}")
        st.code(traceback.format_exc())
        
        # Custom fallback logic to make sure the process doesn't stop
        st.warning("Attempting to recover with fallback options...")
        
        try:
            # Try with smallest llama model
            st.info("Attempting to load tiny-llama model instead...")
            args.redteam_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            args.quant_mode = "8bit"
            asyncio.run(run_conversational_redteam_with_ui(args))
        except Exception as e2:
            st.error(f"Error with tiny-llama model: {{e2}}")
            
            try:
                # Final fallback to template-based generation
                st.info("Using template-based generation as final fallback...")
                from redteamer.red_team.conversational_redteam import ConversationalRedTeam
                
                # Create the engine with templates only
                redteam = ConversationalRedTeam(
                    target_model_type=args.target_type,
                    chatbot_context=args.chatbot_context,
                    redteam_model_id=None,
                    max_iterations=args.max_iterations,
                    verbose=args.verbose
                )
                
                # Set the template flags
                redteam.using_templates = True
                redteam.using_fallback = True
                
                # Run with template fallback
                st.info("Using template-based adversarial prompts - no model required")
                st.info("Continuing with scan...")
                
                # Configure the target model
                model_config = {}
                if args.target_type == "ollama":
                    model_config = {{"model": args.model, "system_prompt": args.system_prompt}}
                elif args.target_type == "curl":
                    model_config = {{"curl_command": args.curl_command, "system_prompt": args.system_prompt}}
                else:
                    model_config = {{"provider": args.target_type, "model": args.model, "system_prompt": args.system_prompt}}
                
                redteam._configure_target_model(model_config)
                results = redteam.run_conversation()
                
                # Show results
                st.success("Completed scan with template-based generation")
                st.json(results)
            except Exception as e3:
                st.error(f"All fallback options failed: {{e3}}")
                st.error("Unable to proceed with the scan. Please try with different parameters.")
    """.format(
        target_type=args.target_type,
        model=args.model,
        system_prompt=repr(args.system_prompt),
        chatbot_context=args.chatbot_context,
        redteam_model_id=repr(args.redteam_model_id),
        max_iterations=args.max_iterations,
        verbose=repr(args.verbose),
        curl_command=repr(args.curl_command)
    )
    
    # Write the temporary script
    temp_script_path = Path("temp_redteam_launcher.py")
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    try:
        # Run the script with streamlit
        print("Launching conversational red teaming tool...")
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