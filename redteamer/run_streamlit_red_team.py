#!/usr/bin/env python
"""
Helper script to launch the Streamlit UI for the conversational red-teaming scanner.
This script simply redirects to the streamlit run command with the proper arguments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the Streamlit UI for the conversational red-teaming scanner"
    )
    
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
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.target_type == "curl" and not args.curl_command:
        print("Error: --curl-command is required for curl targets")
        sys.exit(1)
        
    if args.target_type != "curl" and not args.model:
        print(f"Error: --model is required for {args.target_type} targets")
        sys.exit(1)
    
    # Find the streamlit runner path
    script_dir = Path(__file__).resolve().parent
    streamlit_runner = script_dir / "red_team" / "streamlit_runner.py"
    
    if not streamlit_runner.exists():
        print(f"Error: Streamlit runner not found at {streamlit_runner}")
        sys.exit(1)
    
    # Build the command
    streamlit_args = [
        "streamlit", "run", str(streamlit_runner),
        "--"  # Arguments after this will be passed to the script
    ]
    
    # Add all arguments
    if args.target_type:
        streamlit_args.extend(["--target-type", args.target_type])
        
    if args.context:
        streamlit_args.extend(["--context", args.context])
        
    if args.redteam_model:
        streamlit_args.extend(["--redteam-model", args.redteam_model])
        
    if args.hf_api_key:
        streamlit_args.extend(["--hf-api-key", args.hf_api_key])
        
    if args.model:
        streamlit_args.extend(["--model", args.model])
        
    if args.system_prompt:
        streamlit_args.extend(["--system-prompt", args.system_prompt])
        
    if args.curl_command:
        streamlit_args.extend(["--curl-command", args.curl_command])
        
    if args.iterations:
        streamlit_args.extend(["--iterations", str(args.iterations)])
        
    if args.output_dir:
        streamlit_args.extend(["--output-dir", args.output_dir])
        
    if args.quant_mode:
        streamlit_args.extend(["--quant-mode", args.quant_mode])
        
    if args.verbose:
        streamlit_args.append("--verbose")
    
    # Print the command
    print(f"Launching Streamlit with command: {' '.join(streamlit_args)}")
    
    # Execute the command
    try:
        subprocess.run(streamlit_args)
    except KeyboardInterrupt:
        print("Streamlit process interrupted")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 