#!/usr/bin/env python
"""
Run a demo of the contextual red teaming in sample mode.

This script provides a simple way to run the contextual red teaming demo
without requiring any API keys or real model access. It uses example prompts
and simulated responses to showcase the flow of the red teaming process.

Usage examples:
- Basic usage: python run_demo_sample.py
- Specify number of prompts: python run_demo_sample.py --num-prompts 15
"""

import argparse
import sys
import os

# Add the root directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, root_dir)

# Import after path setup
from redteamer.demo.contextual_demo import run_demo

def main():
    parser = argparse.ArgumentParser(description="Run a sample contextual red teaming demo")
    parser.add_argument("-n", "--num-prompts", type=int, default=10, 
                        help="Number of sample prompts to generate")
    
    args = parser.parse_args()
    
    print(f"Running demo in sample mode with {args.num_prompts} prompts")
    print("Context will be requested through the Streamlit interface")
    
    # Run the demo in sample mode
    run_demo(num_prompts=args.num_prompts, sample_mode=True)

if __name__ == "__main__":
    main() 