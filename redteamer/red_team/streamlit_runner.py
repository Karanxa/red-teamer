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
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

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
    
    # Display configuration
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
        verbose=args.verbose
    )
    
    try:
        # Load the model
        display_model_loading_status("Loading red-teaming model...")
        redteam._load_local_model()
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
        
        # Start pre-generating the first prompt in parallel
        redteam.next_prompt_future = asyncio.create_task(
            redteam._generate_next_prompt_async(redteam.chatbot_context, redteam.conversation_history)
        )
        
        # Run the conversation for the specified number of iterations
        for i in range(redteam.max_iterations):
            # Update metrics
            display_metrics(
                iteration=i,
                max_iterations=redteam.max_iterations,
                vulnerabilities=redteam.vulnerabilities,
                start_time=results["start_time"]
            )
            
            # Get the next prompt (either from the future or generate a new one)
            if redteam.next_prompt_future:
                prompt = await redteam.next_prompt_future
            else:
                prompt = redteam._generate_adversarial_prompt(
                    redteam.chatbot_context, 
                    redteam.conversation_history
                )
            
            # Start generating the next prompt in parallel for the next iteration
            redteam.next_prompt_future = asyncio.create_task(
                redteam._generate_next_prompt_async(
                    redteam.chatbot_context, 
                    redteam.conversation_history
                )
            )
            
            # Send the prompt to the target model
            st.session_state.model_status.info(f"Sending prompt {i+1}/{redteam.max_iterations} to target model...")
            response = redteam._send_to_target_model(prompt, model_config)
            
            # Analyze the response
            analysis = redteam._analyze_response(prompt, response)
            
            # Add to conversation history
            conversation_entry = {
                "iteration": i + 1,
                "prompt": prompt,
                "response": response,
                "analysis": analysis,
                "timestamp": time.time()
            }
            
            redteam.conversation_history.append({
                "prompt": prompt,
                "response": response
            })
            
            results["conversation"].append(conversation_entry)
            
            # Update vulnerabilities if found
            if analysis["potential_vulnerability"]:
                redteam.vulnerabilities.append(analysis)
                results["vulnerabilities"] = redteam.vulnerabilities
            
            # Update UI
            display_conversation(results["conversation"])
            display_vulnerabilities(redteam.vulnerabilities)
            display_metrics(
                iteration=i + 1,
                max_iterations=redteam.max_iterations,
                vulnerabilities=redteam.vulnerabilities,
                start_time=results["start_time"]
            )
            
            # Sleep briefly to allow UI updates
            await asyncio.sleep(0.1)
        
        # Cancel any pending prompt generation
        if redteam.next_prompt_future:
            redteam.next_prompt_future.cancel()
        
        # Compile the final results
        results["vulnerabilities"] = redteam.vulnerabilities
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        # Generate summary statistics
        results["summary"] = {
            "total_iterations": redteam.max_iterations,
            "total_vulnerabilities": len(redteam.vulnerabilities),
            "vulnerability_rate": len(redteam.vulnerabilities) / redteam.max_iterations if redteam.max_iterations > 0 else 0,
            "classifications": {
                "appropriate_refusal": sum(1 for entry in results["conversation"] if entry["analysis"]["classification"] == "appropriate_refusal"),
                "partial_compliance": sum(1 for entry in results["conversation"] if entry["analysis"]["classification"] == "partial_compliance"),
                "full_compliance": sum(1 for entry in results["conversation"] if entry["analysis"]["classification"] == "full_compliance"),
                "error_handling": sum(1 for entry in results["conversation"] if entry["analysis"]["classification"] == "error_handling")
            }
        }
        
        # Save results to file
        output_file = os.path.join(
            args.output_dir,
            f"conv_redteam_{args.target_type}_{int(time.time())}.json"
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        results["output_file"] = output_file
        
        # Update final UI
        display_model_loading_status("Conversational red-teaming completed!")
        display_summary(results)
        
        # Update progress to complete
        st.session_state.progress.progress(1.0)
        
    except Exception as e:
        st.error(f"Error in conversational red-teaming: {str(e)}")
        if args.verbose:
            import traceback
            st.code(traceback.format_exc())

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Conversational Red-Teaming Scanner")
    
    parser.add_argument("--target-type", "-t", required=True,
                        choices=["curl", "openai", "gemini", "huggingface", "ollama"],
                        help="Target model type")
    
    parser.add_argument("--chatbot-context", "-c", required=True,
                        help="Description of the chatbot's purpose, usage, and development reasons")
    
    parser.add_argument("--redteam-model-id", "-r", 
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
    
    parser.add_argument("--max-iterations", "-i", type=int, default=10,
                        help="Maximum number of conversation iterations")
    
    parser.add_argument("--output-dir", "-o", default="results/conversational",
                        help="Directory to save results")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate arguments
    if args.target_type == "curl" and not args.curl_command:
        print("Error: --curl-command is required for curl target type")
        sys.exit(1)
    
    if args.target_type != "curl" and not args.model:
        print(f"Error: --model is required for {args.target_type} target type")
        sys.exit(1)
    
    # Run the Streamlit app with arguments
    asyncio.run(run_conversational_redteam_with_ui(args)) 