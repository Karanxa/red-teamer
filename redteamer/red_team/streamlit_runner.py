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
    
    # Display if fallback mode is enabled
    if hasattr(args, 'fallback_mode') and args.fallback_mode:
        st.warning("⚠️ **FALLBACK MODE ENABLED**: Using template-based generation only. No models will be loaded.", icon="⚠️")
        st.info("This mode bypasses all model loading issues by using predefined templates for adversarial prompts.")
    
    # Display device information
    if hasattr(args, 'device'):
        if args.device == "cpu":
            st.info(f"🖥️ Running in CPU-only mode. Model loading may take longer.", icon="🖥️")
        else:
            st.info(f"🖥️ Device set to: {args.device.upper()}", icon="🖥️")
    
    # Display HuggingFace API key status
    if hasattr(args, 'hf_api_key') and args.hf_api_key:
        st.success("🔑 HuggingFace API key provided - can access gated models", icon="🔑")
    
    def display_model_loading_status(message):
        """Display a loading status message in the Streamlit UI."""
        if not hasattr(run_conversational_redteam_with_ui, 'status_placeholder'):
            run_conversational_redteam_with_ui.status_placeholder = st.empty()
        
        run_conversational_redteam_with_ui.status_placeholder.info(f"🔄 {message}", icon="🔄")
    
    # Ensure we have a valid model ID if not using fallback mode
    if not hasattr(args, 'fallback_mode') or not args.fallback_mode:
        if not hasattr(args, 'redteam_model_id') or not args.redteam_model_id:
            # Set a default model ID that's small enough to work on most systems
            default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            display_model_loading_status(f"No model specified, using default model: {default_model}")
            args.redteam_model_id = default_model
    
    # Update UI with the selected model info
    if hasattr(args, 'redteam_model_id') and args.redteam_model_id:
        st.info(f"🤖 Model: {args.redteam_model_id}")
    
    try:
        # Create the red teaming object with validated model ID
        display_model_loading_status("Initializing red teaming framework...")
        redteam = ConversationalRedTeam(
            target_model_type=args.target_type,
            chatbot_context=args.chatbot_context,
            redteam_model_id=args.redteam_model_id,
            hf_api_key=getattr(args, 'hf_api_key', None),
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            quant_mode=args.quant_mode,
            fallback_mode=args.fallback_mode,
            device=args.device
        )
        
        # Load the model
        display_model_loading_status("Loading red-teaming model...")
        
        # Check if fallback mode is enabled
        if hasattr(args, 'fallback_mode') and args.fallback_mode:
            display_model_loading_status("Fallback mode enabled - skipping model loading")
            redteam.using_templates = True
            redteam.using_fallback = True
            redteam.redteam_model = None
            redteam.redteam_tokenizer = None
        else:
            # Show device information
            if hasattr(args, 'device') and args.device == "cpu":
                display_model_loading_status(f"Loading model in CPU-only mode... (this may take longer)")
            elif hasattr(args, 'quant_mode') and args.quant_mode == "cpu":
                display_model_loading_status(f"Loading model in CPU-only mode... (this may take longer)")
                
            # Try to load the model
            try:
                if hasattr(args, 'device') and args.device == "cpu":
                    # Use CPU-specific loading
                    display_model_loading_status("Using CPU-specific loading method...")
                    if redteam._load_direct_cpu():
                        display_model_loading_status("✅ Model loaded successfully with CPU-specific method")
                    else:
                        display_model_loading_status("❌ CPU-specific loading failed, trying standard loading...")
                        if redteam._load_local_model():
                            display_model_loading_status("✅ Model loaded successfully with standard method")
                        else:
                            display_model_loading_status("❌ All model loading attempts failed")
                            display_model_loading_status("Falling back to template-based generation")
                            redteam.using_templates = True
                else:
                    # Use standard loading with auto device map
                    display_model_loading_status("Using standard loading method with auto device mapping...")
                    if redteam._load_local_model():
                        display_model_loading_status("✅ Model loaded successfully")
                    else:
                        display_model_loading_status("❌ Standard loading failed, trying CPU-specific method...")
                        if redteam._load_direct_cpu():
                            display_model_loading_status("✅ Model loaded with CPU-specific method")
                        else:
                            display_model_loading_status("❌ All model loading attempts failed")
                            display_model_loading_status("Falling back to template-based generation")
                            redteam.using_templates = True
                
                # Check model loading status one more time
                if not redteam.check_model_loaded() and not redteam.using_templates and not redteam.using_fallback:
                    # If still not loaded, force template mode
                    display_model_loading_status("⚠️ Model did not load properly. Forcing template mode.")
                    redteam.using_templates = True
                
                # Display final model status
                model_status = redteam.get_model_status()
                display_model_loading_status(f"Final status: {model_status}")
                
            except ImportError as import_error:
                # Special handling for import errors
                error_message = str(import_error)
                display_model_loading_status(f"❌ Import error: {error_message}")
                
                if "huggingface_hub" in error_message and "split_torch_state_dict_into_shards" in error_message:
                    display_model_loading_status("⚠️ This is a known issue with huggingface_hub package version.")
                    display_model_loading_status("Try updating it with: pip install --upgrade huggingface_hub>=0.21.0")
                elif "bitsandbytes" in error_message:
                    display_model_loading_status("⚠️ BitsAndBytes library not installed. Install with: pip install bitsandbytes")
                elif "transformers" in error_message:
                    display_model_loading_status("⚠️ Transformers library issue. Install with: pip install --upgrade transformers")
                
                display_model_loading_status("Falling back to template-based generation")
                redteam.using_templates = True
                
            except Exception as model_error:
                display_model_loading_status(f"❌ Error loading model: {str(model_error)}")
                display_model_loading_status("Falling back to template-based generation")
                redteam.using_templates = True
        
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

def main():
    """Run the conversational red teamer tool."""
    parser = argparse.ArgumentParser(description="Red Teaming Tool with UI")
    parser.add_argument("--target-type", default="ollama", help="Target model type")
    parser.add_argument("--model", default="llama3", help="Model name")
    parser.add_argument("--system-prompt", help="System prompt for the target model")
    parser.add_argument("--chatbot-context", default="You are a helpful AI assistant.", help="Description of the chatbot being tested")
    parser.add_argument("--redteam-model-id", help="Model ID for red teaming")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quant-mode", default="auto", help="Quantization mode")
    parser.add_argument("--fallback-mode", action="store_true", help="Enable fallback mode")
    parser.add_argument("--device", default="gpu", help="Device to run the model on")
    parser.add_argument("--hf-api-key", help="HuggingFace API key for accessing gated models")
    
    args = parser.parse_args()
    asyncio.run(run_conversational_redteam_with_ui(args))

if __name__ == "__main__":
    main() 