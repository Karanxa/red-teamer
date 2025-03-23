#!/usr/bin/env python
"""
Example script demonstrating how to test a single model with a prompt.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Optional

# Add parent directory to path to import redteamer module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redteamer.main import test_model, evaluate_response, setup_logging

def main():
    """Main function to test a model with a prompt and evaluate the response."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test a model with a prompt')
    parser.add_argument('--provider', '-p', required=True, help='Model provider (e.g., openai, anthropic)')
    parser.add_argument('--model', '-m', required=True, help='Model name')
    parser.add_argument('--prompt', required=True, help='Prompt to send to the model')
    parser.add_argument('--system-prompt', help='Optional system prompt')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY', 
                        help='Environment variable containing the API key')
    parser.add_argument('--temperature', type=float, default=0.7, help='Model temperature')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens to generate')
    parser.add_argument('--vector-file', help='Path to attack vector JSON file for evaluation')
    parser.add_argument('--evaluator-type', default='rule-based', 
                        choices=['rule-based', 'model-based', 'hybrid'],
                        help='Type of evaluator to use')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Create model configuration
    model_config = {
        'name': args.model,
        'provider': args.provider,
        'parameters': {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens
        },
        'api_key_env': args.api_key_env
    }
    
    # Test model
    print(f"Testing model: {args.provider}/{args.model}")
    print(f"Prompt: {args.prompt}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    
    # Get response
    response = test_model(model_config, args.prompt, args.system_prompt)
    
    # Print response
    print("\nModel Response:")
    print("-" * 80)
    print(response.get('response_text', ''))
    print("-" * 80)
    print(f"Tokens: {response.get('tokens', 0)}")
    print(f"Latency: {response.get('latency', 0):.3f} seconds")
    
    # Evaluate response if a vector file is provided
    if args.vector_file:
        # Load attack vector
        try:
            with open(args.vector_file, 'r') as f:
                attack_vector_data = json.load(f)
            
            # For simplicity, if the file has a 'vectors' array, use the first vector
            if 'vectors' in attack_vector_data and len(attack_vector_data['vectors']) > 0:
                attack_vector = attack_vector_data['vectors'][0]
            else:
                attack_vector = attack_vector_data
            
            # Evaluate the response
            print("\nEvaluating response...")
            
            # For model-based or hybrid evaluation, we'll use the same model config
            evaluator_model_config = model_config.copy() if args.evaluator_type != 'rule-based' else None
            
            evaluation = evaluate_response(
                response_text=response.get('response_text', ''),
                attack_vector=attack_vector,
                evaluator_type=args.evaluator_type,
                model_config=evaluator_model_config
            )
            
            # Print evaluation results
            print("\nEvaluation Results:")
            print(f"Success: {evaluation.get('success', False)} (Attack {'succeeded' if evaluation.get('success', False) else 'failed'})")
            print(f"Confidence: {evaluation.get('confidence', 0):.2f}")
            print(f"Explanation: {evaluation.get('explanation', '')}")
            
            if 'matched_rules' in evaluation and evaluation['matched_rules']:
                print(f"Matched rules: {', '.join(evaluation['matched_rules'])}")
        
        except Exception as e:
            print(f"Error evaluating response: {e}")

if __name__ == "__main__":
    main() 