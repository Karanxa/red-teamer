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
import asyncio
import logging
from pathlib import Path
import streamlit as st
import json

# Make this script runnable from any location
script_dir = Path(__file__).parent
os.chdir(script_dir.parent)

# Import the streamlit runner
from redteamer.red_team.streamlit_runner import run_conversational_redteam_with_ui

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the conversational red teaming application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational LLM Red Teaming")
    parser.add_argument("--target-type", default="ollama", help="Target model type (ollama, openai, anthropic, etc.)")
    parser.add_argument("--model", default="gemma3:1b", help="Model name")
    parser.add_argument("--system-prompt", help="System prompt for the target model")
    parser.add_argument("--chatbot-context", default="You are a helpful AI assistant.", help="Description of the chatbot being tested")
    parser.add_argument("--redteam-model-id", help="Model ID for red teaming (if not provided, will use default)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of conversation iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--curl-command", help="Custom curl command for target model")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", 
                       help="Device to run the model on: 'cpu' (slower but works everywhere) or 'gpu' (faster but requires compatible hardware)")
    parser.add_argument("--fallback-mode", action="store_true", 
                       help="Use template-based generation only, without attempting to load models (useful when dependencies have issues)")
    parser.add_argument("--hf-api-key", help="HuggingFace API key for accessing gated models")
    
    args = parser.parse_args()
    
    # Print configuration summary
    print("\nCommand Line - Simpler, text-based output without browser")
    print("\n\nConversational Red Teaming Configuration:")
    print(f"Provider: {args.target_type}")
    print(f"Model: {args.model}")
    print(f"Chatbot Context: {args.chatbot_context}")
    print(f"Red Teaming Model: {args.redteam_model_id if args.redteam_model_id else '(using default)'}")
    print(f"Maximum Iterations: {args.max_iterations}")
    print(f"Interface: Streamlit UI")
    print(f"Verbose: {'Yes' if args.verbose else 'No'}")
    print(f"HuggingFace API Key: {'Provided' if args.hf_api_key else 'Not provided'}")
    print("")
    
    # If no red team model is specified, ask the user
    if args.redteam_model_id is None:
        redteam_options = {
            "1": "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Recommended for CPU)",
            "2": "meta-llama/Llama-2-7b-chat-hf (Requires GPU)",
            "3": "gpt2 (Small model, works on most hardware)",
            "4": "distilgpt2 (Ultra-small model for limited hardware)",
            "5": "None - Use template-based generation (No model required)"
        }
        
        print("\nPlease select a red teaming model:")
        for key, value in redteam_options.items():
            print(f"{key}: {value}")
        
        model_choice = input("\nEnter your choice (1-5) or provide a custom model ID: ")
        
        if model_choice in redteam_options:
            if model_choice == "5":
                args.redteam_model_id = None
            else:
                args.redteam_model_id = redteam_options[model_choice].split(" ")[0]
        else:
            if model_choice.strip():  # If non-empty
                args.redteam_model_id = model_choice.strip()
            else:
                args.redteam_model_id = None
        
        print(f"\nSelected: {args.redteam_model_id if args.redteam_model_id else 'Template-based generation'}")
    
    # If using a HuggingFace model that might require an API key, ask for it
    if args.redteam_model_id and "/" in args.redteam_model_id and not args.hf_api_key and not args.fallback_mode:
        print("\nSome models from HuggingFace require an API key to access.")
        api_key_input = input("Enter your HuggingFace API key (or press Enter to skip): ")
        if api_key_input.strip():
            args.hf_api_key = api_key_input.strip()
            print("HuggingFace API key set.")
            
            # Set environment variable for the current process
            import os
            os.environ['HUGGINGFACE_TOKEN'] = args.hf_api_key
            os.environ['HF_TOKEN'] = args.hf_api_key
    
    # Display information about the selected device
    if args.device == "cpu":
        print("\n📌 CPU mode selected - slower but more compatible.")
        print("   - Models will run in CPU-optimized mode")
        print("   - Larger models may not work due to memory constraints")
    else:  # gpu
        print("\n📌 GPU mode selected - faster but requires compatible hardware.")
        print("   - Will use GPU acceleration if available")
        print("   - Will fall back to CPU if no compatible GPU is found")
    
    # Display information about fallback mode if enabled
    if args.fallback_mode:
        print("\n⚠️ Fallback mode enabled - will use templated adversarial prompts only.")
        print("   - No model loading will be attempted")
        print("   - This avoids dependency issues with transformers/accelerate")
    
    # Try to import bitsandbytes if using CPU mode
    if args.device == "cpu" and not args.fallback_mode:
        try:
            import bitsandbytes
            print("✅ Found bitsandbytes library for CPU quantization")
        except ImportError:
            print("⚠️  Warning: bitsandbytes library not found.")
            print("   For better CPU performance, install it with: pip install bitsandbytes")
    
    # Ask if the user is ready to launch
    launch_choice = input("\n[?] Ready to launch the Streamlit interface for conversational red teaming? (Y/n): ")
    if launch_choice.lower() in ["n", "no"]:
        print("Exiting...")
        return
    
    print("\n\nLaunching Streamlit Interface...")
    print("The red teaming process will run in the Streamlit interface that is being launched.")
    # Prepare streamlit UI script
    streamlit_ui_code = """
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

# Import the streamlit fix and apply it
from redteamer.utils.streamlit_fix import apply_torch_streamlit_fix
apply_torch_streamlit_fix()

# Import the streamlit runner
from redteamer.red_team.streamlit_runner import run_conversational_redteam_with_ui
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the target model config
def get_target_model_config(args):
    model_config = {}
    if args.target_type == "ollama":
        model_config = {"model": args.model, "system_prompt": args.system_prompt}
    elif args.target_type == "curl":
        model_config = {"curl_command": args.curl_command, "system_prompt": args.system_prompt}
    else:
        model_config = {"provider": args.target_type, "model": args.model, "system_prompt": args.system_prompt}
    return model_config

# Custom function to display conversation as a chat
def display_conversation_as_chat(conversation_history):
    if not conversation_history:
        return
        
    st.subheader("Conversation Between Red Team Agent and Target Model")
    
    for i, exchange in enumerate(conversation_history):
        # Red team prompt
        with st.chat_message("assistant", avatar="🔴"):
            st.markdown(f"**Red Team Agent**: {exchange.get('prompt', '')}")
            
        # Target model response
        with st.chat_message("user", avatar="🎯"):
            st.markdown(f"**Target Model**: {exchange.get('response', '')}")
        
        # Show analysis if available
        if 'analysis' in exchange and exchange['analysis']:
            with st.expander(f"Analysis of Exchange {i+1}"):
                st.markdown(exchange['analysis'])
                
                # Show vulnerability details if found
                if exchange.get('found_vulnerability', False):
                    st.warning("⚠️ **Potential vulnerability detected in this exchange!**")
                    st.markdown(f"**Type**: {exchange.get('vulnerability_type', 'Unknown')}")
                    st.markdown(f"**Severity**: {exchange.get('severity', 'Unknown')}")

# Run the conversational red teaming
if __name__ == "__main__":
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Conversational Red Teaming",
        page_icon="🔴",
        layout="wide"
    )
    
    # Header
    st.title("🔴 Conversational Red Teaming")
    st.markdown(\"\"\"
    This tool simulates adversarial conversations between a red teaming agent and your target model.
    The red teaming agent attempts to elicit harmful, incorrect, or undesired responses.
    \"\"\")
    
    # Parse command line args
    import json
    
    # Get args from environment variable
    args_json = os.environ.get('REDTEAM_ARGS', '{}')
    args_dict = json.loads(args_json)
    
    class Args:
        def __init__(self, args_dict):
            self.target_type = args_dict.get('target_type', 'ollama')
            self.model = args_dict.get('model', 'gemma3:1b')
            self.system_prompt = args_dict.get('system_prompt')
            self.chatbot_context = args_dict.get('chatbot_context', 'You are a helpful AI assistant.')
            self.redteam_model_id = args_dict.get('redteam_model_id')
            self.max_iterations = int(args_dict.get('max_iterations', 10))
            self.verbose = bool(args_dict.get('verbose', False))
            self.curl_command = args_dict.get('curl_command')
            # Get device preference directly from args
            self.device = args_dict.get('device', 'gpu')
            self.hf_api_key = args_dict.get('hf_api_key')
            self.output_dir = "results/conversational"
            # If CPU mode selected, force CPU quantization
            self.quant_mode = "cpu" if self.device == "cpu" else "auto"
            self.fallback_mode = bool(args_dict.get('fallback_mode', False))
    
    args = Args(args_dict)
    
    try:
        # Show the configuration
        with st.expander("Configuration"):
            st.markdown(f"**Target Model**: {args.target_type} / {args.model}")
            st.markdown(f"**Chatbot Context**: {args.chatbot_context}")
            st.markdown(f"**Red Team Model**: {args.redteam_model_id if args.redteam_model_id else 'Default'}")
            st.markdown(f"**Max Iterations**: {args.max_iterations}")
            st.markdown(f"**Device Mode**: {'CPU only' if args.device == 'cpu' else 'GPU when available'}")
            st.markdown(f"**Quantization**: {args.quant_mode}")
            st.markdown(f"**HuggingFace API Key**: {'Provided ✅' if args.hf_api_key else 'Not provided ❌'}")
            if args.fallback_mode:
                st.markdown("**Fallback Mode**: Enabled ⚠️ (using template-based generation only)")
        
        # Create progress bar and status
        progress = st.progress(0)
        status = st.empty()
        
        # Add conversation containers
        conversation_container = st.container()
        analysis_container = st.container()
        
        # Placeholder for conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            st.session_state.vulnerabilities_found = 0
        
        # Function to update UI with conversation progress
        def update_ui_callback(iteration, total, current_exchange, vulnerabilities_found=0):
            progress.progress(iteration / total)
            status.markdown(f"**Status**: Running iteration {iteration}/{total}. Found {vulnerabilities_found} vulnerabilities so far.")
            
            # Update conversation history
            if current_exchange and 'prompt' in current_exchange and 'response' in current_exchange:
                st.session_state.conversation_history.append(current_exchange)
                st.session_state.vulnerabilities_found = vulnerabilities_found
                
                # Display updated conversation
                with conversation_container:
                    display_conversation_as_chat(st.session_state.conversation_history)
        
        # Properly await the coroutine using asyncio.run
        status.markdown("**Status**: Loading red-teaming model...")
        results = asyncio.run(run_conversational_redteam_with_ui(args, ui_callback=update_ui_callback))
        
        # Progress complete
        progress.progress(100)
        status.markdown(f"**Status**: Completed! Found {st.session_state.vulnerabilities_found} vulnerabilities.")
        
        # Display final results
        with analysis_container:
            st.subheader("Vulnerability Summary")
            
            if 'vulnerabilities' in results and results['vulnerabilities']:
                for i, vuln in enumerate(results['vulnerabilities']):
                    with st.expander(f"Vulnerability #{i+1}: {vuln.get('vulnerability_type', 'Unknown')}"):
                        st.markdown(f"**Prompt**: {vuln.get('prompt', '')}")
                        st.markdown(f"**Response**: {vuln.get('response', '')}")
                        st.markdown(f"**Analysis**: {vuln.get('analysis', '')}")
                        st.markdown(f"**Severity**: {vuln.get('severity', 'Unknown')}")
            else:
                st.success("No vulnerabilities found. Your model appears to be robust against the tested attacks.")
            
    except Exception as e:
        import traceback
        
        st.error("Error: " + str(e))
        st.code(traceback.format_exc())
        
        # Custom fallback logic to make sure the process doesn't stop
        st.warning("Attempting to recover with fallback options...")
        
        try:
            # Try with smallest llama model
            st.info("Attempting to load tiny-llama model instead...")
            args.redteam_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # If we have a HuggingFace API key, make sure it's set
            if hasattr(args, 'hf_api_key') and args.hf_api_key:
                st.info("Using provided HuggingFace API key for model access")
                # Set the environment variables for the Hugging Face API
                import os
                os.environ["HUGGINGFACE_TOKEN"] = args.hf_api_key
                os.environ["HF_TOKEN"] = args.hf_api_key
            
            # Set appropriate quantization based on device
            if args.device == "cpu":
                st.info("Using CPU-optimized loading for TinyLlama")
                args.quant_mode = "cpu"
            else:
                st.info("Using quantized model with automatic device placement")
                args.quant_mode = "8bit"
            
            # Wrap in try-except to handle the set_page_config error
            try:
                results = asyncio.run(run_conversational_redteam_with_ui(args, ui_callback=update_ui_callback))
                
                # Display conversation if available
                if hasattr(st.session_state, 'conversation_history') and st.session_state.conversation_history:
                    with conversation_container:
                        display_conversation_as_chat(st.session_state.conversation_history)
            except Exception as page_config_error:
                if "set_page_config() can only be called once" in str(page_config_error):
                    st.warning("Streamlit page already configured. Continuing with fallback...")
                else:
                    raise
        except Exception as e2:
            st.error("Error with tiny-llama model: " + str(e2))
            
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
                
                # Initialize the logger as an instance variable since it's missing
                redteam.logger = logging.getLogger('redteam')
                
                # Set the template flags
                redteam.using_templates = True
                redteam.using_fallback = True
                
                # Check if CPU mode is selected and bitsandbytes is not installed
                if args.device == "cpu":
                    try:
                        import bitsandbytes
                        st.success("Found bitsandbytes library for CPU quantization")
                    except ImportError:
                        st.warning("bitsandbytes library not found. For better CPU performance, install it with: pip install bitsandbytes")
                
                # Run with template fallback
                st.info("Using template-based adversarial prompts - no model required")
                st.info("Continuing with scan...")
                
                # Configure the target model
                model_config = get_target_model_config(args)
                
                # Configure target model
                redteam._configure_target_model(model_config)
                
                # Create conversation history for UI
                conversation_history = []
                
                # Update progress function
                def template_update_ui(iteration, total, exchange, vulnerabilities_found=0):
                    progress.progress(iteration / total)
                    status.markdown(f"**Status**: Running iteration {iteration}/{total} with template-based generation.")
                    
                    if exchange:
                        conversation_history.append(exchange)
                        with conversation_container:
                            display_conversation_as_chat(conversation_history)
                
                # Run the conversation with UI updates
                results = asyncio.run(redteam.run_redteam_conversation(
                    model_config, 
                    ui_callback=template_update_ui
                ))
                
                # Display final conversation
                with conversation_container:
                    display_conversation_as_chat(conversation_history)
                
                # Show results
                st.success("Completed scan with template-based generation")
                
                # Display results analysis
                with analysis_container:
                    st.subheader("Vulnerability Summary")
                    
                    if 'vulnerabilities' in results and results['vulnerabilities']:
                        for i, vuln in enumerate(results['vulnerabilities']):
                            with st.expander(f"Vulnerability #{i+1}: {vuln.get('vulnerability_type', 'Unknown')}"):
                                st.markdown(f"**Prompt**: {vuln.get('prompt', '')}")
                                st.markdown(f"**Response**: {vuln.get('response', '')}")
                                st.markdown(f"**Analysis**: {vuln.get('analysis', '')}")
                                st.markdown(f"**Severity**: {vuln.get('severity', 'Unknown')}")
                    else:
                        st.success("No vulnerabilities found. Your model appears to be robust against the tested attacks.")
            except Exception as e3:
                st.error("All fallback options failed: " + str(e3))
                st.error("Unable to proceed with the scan. Please try with different parameters.")
"""

    # Write the streamlit script to a temporary file
    temp_script_path = Path("temp_redteam_launcher.py")
    with open(temp_script_path, "w") as f:
        f.write(streamlit_ui_code)
    
    # Pass the command line args to the script via environment variable
    args_dict = {
        'target_type': args.target_type,
        'model': args.model,
        'system_prompt': args.system_prompt,
        'chatbot_context': args.chatbot_context,
        'redteam_model_id': args.redteam_model_id,
        'max_iterations': args.max_iterations,
        'verbose': args.verbose,
        'curl_command': args.curl_command,
        'device': args.device,
        'fallback_mode': args.fallback_mode,
        'hf_api_key': args.hf_api_key
    }
    
    env = os.environ.copy()
    env['REDTEAMER_FIX_STREAMLIT'] = "1"
    env['REDTEAM_ARGS'] = json.dumps(args_dict)
    
    try:
        # Run the script with streamlit
        print("Launching conversational red teaming tool...")
        subprocess.run(["streamlit", "run", str(temp_script_path)], env=env, check=True)
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