#!/usr/bin/env python3
"""
Simple wrapper to run the contextual red teaming demo in sample mode.
This script doesn't require any API keys and uses pre-generated example data.

Usage examples:
    # Basic usage (will prompt for context)
    python run_demo_sample.py
    
    # Specify number of prompts
    python run_demo_sample.py --num-prompts 15
    
    # Provide chatbot context directly
    python run_demo_sample.py --context "A financial advisor chatbot for retirement planning"
    
    # Combine options
    python run_demo_sample.py -n 20 -c "A healthcare chatbot for patient scheduling"
"""

import sys
import argparse

def main():
    """Run the demo in sample mode"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Run a demo of contextual red teaming with sample data")
        parser.add_argument("--num-prompts", "-n", type=int, default=10, 
                            help="Number of sample prompts to generate (default: 10)")
        parser.add_argument("--context", "-c", type=str, default="",
                            help="Context description for the chatbot (will prompt if not provided)")
        
        args = parser.parse_args()
        
        from redteamer.demo.contextual_demo import run_demo
        
        # Run the demo in sample mode with provided arguments
        print(f"Starting demo in sample mode with {args.num_prompts} example prompts...")
        if args.context:
            print(f"Using provided context: '{args.context}'")
        
        run_demo(sample_mode=True, num_prompts=args.num_prompts, context_file=None)
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you have installed the redteamer package and its dependencies.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 