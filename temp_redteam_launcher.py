
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
    st.markdown("""
    This tool simulates adversarial conversations between a red teaming agent and your target model.
    The red teaming agent attempts to elicit harmful, incorrect, or undesired responses.
    """)
    
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
