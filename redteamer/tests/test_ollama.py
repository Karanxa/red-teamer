#!/usr/bin/env python3
"""
Test script for the OllamaConnector with streaming support.
This script verifies that our connector properly handles Ollama's streaming API responses.
"""

import os
import sys
import json
import logging
from typing import Dict

# Add the parent directory to the path to import from redteamer
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, parent_dir)

from redteamer.utils.model_connector import OllamaConnector, ModelConnector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ollama_test")

def test_ollama_streaming(model_name: str = "gemma3:1b"):
    """Test the Ollama streaming connector."""
    logger.info(f"Testing Ollama streaming connector with model: {model_name}")
    
    # Create an Ollama connector
    connector = OllamaConnector()
    
    # Test a simple prompt
    prompt = "How are you today?"
    logger.info(f"Sending prompt: {prompt}")
    
    # Try the streaming method
    logger.info("Using streaming method...")
    result = connector.generate_completion_streaming(
        model_name=model_name,
        prompt=prompt
    )
    
    # Display the results
    logger.info("Streaming response complete:")
    logger.info(f"Response text: {result['response_text']}")
    logger.info(f"Token count: {result['token_count']}")
    logger.info(f"Latency: {result['latency']:.2f} seconds")
    
    # Check if we have response data (metadata)
    if 'response_data' in result and result['response_data']:
        logger.info("Response metadata:")
        logger.info(f"  Prompt tokens: {result['response_data'].get('prompt_eval_count', 'N/A')}")
        logger.info(f"  Completion tokens: {result['response_data'].get('eval_count', 'N/A')}")
        logger.info(f"  Total duration: {result['response_data'].get('total_duration', 'N/A')}")
    
    # Save the result to a JSON file
    with open("ollama_streaming_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Results saved to ollama_streaming_result.json")
    
    return result

def test_model_connector(model_name: str = "gemma3:1b"):
    """Test the Ollama connector through the ModelConnector interface."""
    logger.info(f"Testing ModelConnector with Ollama model: {model_name}")
    
    # Create a model connector
    connector = ModelConnector()
    
    # Configure the model
    model_config = {
        "provider": "ollama",
        "model_id": model_name,
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    # Test a simple prompt
    prompt = "What's the capital of France?"
    logger.info(f"Sending prompt: {prompt}")
    
    # Generate a completion
    result = connector.generate_completion(
        model_config=model_config,
        prompt=prompt
    )
    
    # Display the results
    logger.info("Response complete:")
    logger.info(f"Response text: {result['response_text']}")
    logger.info(f"Token count: {result.get('token_count', {})}")
    logger.info(f"Latency: {result['latency_ms']/1000:.2f} seconds")
    
    # Save the result to a JSON file
    with open("ollama_connector_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Results saved to ollama_connector_result.json")
    
    return result

if __name__ == "__main__":
    # Check if a model name was provided as an argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma3:1b"
    
    # Test both methods
    logger.info("-" * 50)
    logger.info("TESTING OLLAMA STREAMING CONNECTOR")
    logger.info("-" * 50)
    streaming_result = test_ollama_streaming(model_name)
    
    logger.info("\n" + "-" * 50)
    logger.info("TESTING MODEL CONNECTOR WITH OLLAMA")
    logger.info("-" * 50)
    connector_result = test_model_connector(model_name)
    
    logger.info("\n" + "-" * 50)
    logger.info("TESTS COMPLETE")
    logger.info("-" * 50) 