"""
Streamlit UI for the Conversational Red-Teaming Scanner.

This module provides a streamlit-based interface for monitoring and interacting
with the conversational red-teaming process in real-time.
"""

import os
import sys
import time
import json
import asyncio
import threading
import argparse
from typing import Dict, List, Optional, Any, Callable

# Import only when the file is run directly through Streamlit
# Do not import at module level to avoid warnings
def get_streamlit():
    """Get the streamlit module only when needed in the Streamlit context."""
    try:
        import streamlit as st
        return st
    except (ImportError, RuntimeError):
        print("Streamlit not available or not running in a Streamlit context")
        return None

# Initialize Streamlit app when the module is run directly
def main():
    """Main entry point when running as a Streamlit app."""
    st = get_streamlit()
    if not st:
        print("This script must be run using 'streamlit run'")
        sys.exit(1)
    
    # Get command line arguments
    args = parse_arguments()
    
    # Check if required parameters are provided
    check_required_args(args)
    
    # Add the parent directory to the system path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, parent_dir)
    
    # Now import required modules
    from redteamer.red_team.conversational_redteam import ConversationalRedTeam
    
    # Initialize the app
    initialize_streamlit_app(st)
    
    # Run the conversational red-teaming process
    run_conversational_redteam_with_ui(args, st)

def parse_arguments():
    """
    Parse command line arguments passed to Streamlit.
    
    Returns:
        Parsed arguments
    """
    # When running with streamlit run, command line args come after --
    parser = argparse.ArgumentParser(description="Conversational Red-Teaming Scanner")
    
    parser.add_argument("--target-type", "-t", required=True,
                        choices=["curl", "openai", "gemini", "huggingface", "ollama"],
                        help="Target model type")
    
    parser.add_argument("--context", "-c", required=True,
                        help="Description of the chatbot's purpose, usage, and development reasons")
    
    parser.add_argument("--redteam-model", "-r", 
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Hugging Face model identifier for the red-teaming model")
    
    parser.add_argument("--hf-api-key",
                        help="Hugging Face API key")
    
    parser.add_argument("--model", "-m",
                        help="Target model name (not needed for curl)")
    
    parser.add_argument("--system-prompt", "-s",
                        help="System prompt for the target model")
    
    parser.add_argument("--curl-command",
                        help="Curl command template with {prompt} and optional {system_prompt} placeholders")
    
    parser.add_argument("--iterations", "-i", type=int, default=10,
                        help="Maximum number of conversation iterations")
    
    parser.add_argument("--output-dir", "-o", default="results/conversational",
                        help="Directory to save results")
    
    parser.add_argument("--quant-mode", "-q", default="auto",
                        choices=["auto", "8bit", "4bit", "cpu"],
                        help="Quantization mode for the model")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    # Parse args passed after the -- in streamlit run
    try:
        sys_args = sys.argv[1:]
        # Find the position of -- and take everything after it
        if "--" in sys_args:
            dash_index = sys_args.index("--")
            streamlit_args = sys_args[dash_index+1:]
            return parser.parse_args(streamlit_args)
        else:
            return parser.parse_args()
    except:
        # If argument parsing fails, provide a dummy namespace for development
        args = argparse.Namespace()
        args.target_type = None
        args.context = None
        args.redteam_model = None
        args.model = None
        args.system_prompt = None
        args.curl_command = None
        args.iterations = 10
        args.output_dir = "results/conversational"
        args.verbose = False
        args.quant_mode = "auto"
        args.hf_api_key = None
        return args

def check_required_args(args):
    """Check if required arguments are provided."""
    st = get_streamlit()
    if not st:
        return
    
    missing_args = []
    
    if not args.target_type:
        missing_args.append("target-type")
    
    if not args.context:
        missing_args.append("context")
    
    if args.target_type == "curl" and not args.curl_command:
        missing_args.append("curl-command")
    
    if args.target_type not in ["curl", None] and not args.model:
        missing_args.append("model")
    
    if missing_args:
        st.error(f"Missing required arguments: {', '.join(missing_args)}")
        st.markdown("""
        ### Usage
        To use this application, you need to provide the following arguments:
        
        ```
        streamlit run redteamer/red_team/streamlit_runner.py -- --target-type TYPE --context "DESCRIPTION" [other options]
        ```
        
        For a curl target, you must also provide --curl-command.
        For other targets, you must provide --model.
        """)
        st.stop()

def initialize_streamlit_app(st):
    """
    Initialize the Streamlit app with basic configuration.
    
    Args:
        st: Streamlit module
    """
    st.set_page_config(
        page_title="Conversational Red-Teaming Scanner",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("Conversational Red-Teaming Scanner")
    st.markdown("""
    This tool automatically tests conversational AI models with adversarial prompts.
    The red-teaming model generates prompts iteratively based on the model's responses.
    """)

def display_config(st, args):
    """
    Display the configuration settings.
    
    Args:
        st: Streamlit module
        args: Command line arguments
    """
    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Model Type**")
            st.code(args.target_type)
            
            st.markdown("**Target Model**")
            if args.target_type == "curl":
                st.code("Custom curl command")
            else:
                st.code(args.model)
            
            st.markdown("**Red-Teaming Model**")
            st.code(args.redteam_model)
        
        with col2:
            st.markdown("**Max Iterations**")
            st.code(str(args.iterations))
            
            st.markdown("**Quantization Mode**")
            st.code(args.quant_mode)
            
            st.markdown("**Chatbot Context**")
            st.text_area("", value=args.context, height=100, disabled=True)

async def run_conversational_redteam_with_ui(args, st):
    """
    Run the conversational red-teaming process with Streamlit UI.
    
    Args:
        args: Command line arguments
        st: Streamlit module
    """
    from redteamer.red_team.conversational_redteam import ConversationalRedTeam
    
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
    
    # Display configuration
    display_config(st, args)
    
    # Initialize progress indicators
    progress = st.progress(0)
    st.session_state['progress'] = progress
    
    status = st.empty()
    status.info("Initializing red-teaming process...")
    
    # Create conversation containers
    conversation_container = st.container()
    st.session_state['conversation_container'] = conversation_container
    
    vuln_container = st.container()
    st.session_state['vuln_container'] = vuln_container
    
    metrics_container = st.container()
    st.session_state['metrics_container'] = metrics_container
    
    summary_container = st.container()
    st.session_state['summary_container'] = summary_container
    
    # Create the conversational red-teaming engine
    redteam = ConversationalRedTeam(
        target_model_type=args.target_type,
        chatbot_context=args.context,
        redteam_model_id=args.redteam_model,
        hf_api_key=args.hf_api_key,
        output_dir=args.output_dir,
        max_iterations=args.iterations,
        verbose=args.verbose,
        quant_mode=args.quant_mode
    )
    
    # Run the red-teaming process
    try:
        status.info("Running red-teaming process...")
        results = await redteam.run_redteam_conversation(model_config)
        status.success("Red-teaming process completed successfully!")
        
        # Display summary
        with summary_container:
            st.header("Red-Teaming Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Iterations", results["summary"]["total_iterations"])
                st.metric("Total Vulnerabilities", results["summary"]["total_vulnerabilities"])
                st.metric("Vulnerability Rate", f"{results['summary']['vulnerability_rate']:.1%}")
                
            with col2:
                st.markdown("**Model Loading**")
                st.success("Model loaded successfully") if results["summary"]["model_loaded_successfully"] else st.error("Model loading failed, used fallback prompts")
                
                st.markdown("**Results File**")
                st.code(results.get("output_file", "No output file"))
        
    except Exception as e:
        status.error(f"Error during red-teaming: {str(e)}")
        st.exception(e)

# Only run the app when this file is directly executed by Streamlit
if __name__ == "__main__":
    main() 