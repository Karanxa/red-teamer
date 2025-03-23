#!/usr/bin/env python3
"""
Test script that compares our OllamaConnector with direct curl command execution.
This helps verify that our implementation correctly handles all the data that the raw API provides.
"""

import os
import sys
import json
import time
import logging
import subprocess
from typing import Dict, List, Any

# Add the parent directory to the path to import from redteamer
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, parent_dir)

from redteamer.utils.model_connector import OllamaConnector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ollama_curl_test")

def test_direct_curl(model_name: str = "gemma3:1b", prompt: str = "How are you today?") -> Dict[str, Any]:
    """Test using direct curl command to the Ollama API."""
    logger.info(f"Testing direct curl with model: {model_name}")
    logger.info(f"Prompt: {prompt}")
    
    # Start timing
    start_time = time.time()
    
    # Prepare the curl command
    curl_cmd = [
        "curl", 
        "http://localhost:11434/api/generate", 
        "-d", 
        json.dumps({
            "model": model_name,
            "prompt": prompt
        })
    ]
    
    # Execute the curl command
    process = subprocess.Popen(
        curl_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Process the streaming response
    response_chunks = []
    accumulated_response = ""
    final_metadata = None
    
    # Read from stdout line by line
    for line in process.stdout:
        # Try to parse as JSON
        try:
            chunk = json.loads(line)
            response_chunks.append(chunk)
            
            # Accumulate the response
            if "response" in chunk:
                accumulated_response += chunk["response"]
            
            # Save the final metadata
            if chunk.get("done", False):
                final_metadata = chunk
                
        except json.JSONDecodeError:
            logger.warning(f"Could not parse line as JSON: {line}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare the result dictionary
    result = {
        "prompt": prompt,
        "model": model_name,
        "response_text": accumulated_response,
        "latency": elapsed_time,
        "chunks": response_chunks,
        "chunk_count": len(response_chunks),
        "metadata": final_metadata
    }
    
    # Calculate token counts if available
    if final_metadata:
        result["token_count"] = {
            "prompt": final_metadata.get("prompt_eval_count", 0),
            "completion": final_metadata.get("eval_count", 0),
            "total": final_metadata.get("prompt_eval_count", 0) + final_metadata.get("eval_count", 0)
        }
    
    # Output the results
    logger.info(f"Response: {accumulated_response}")
    logger.info(f"Received {len(response_chunks)} chunks")
    logger.info(f"Latency: {elapsed_time:.2f} seconds")
    
    # Save the result to a JSON file
    with open("ollama_curl_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Results saved to ollama_curl_result.json")
    
    return result

def test_comparison(model_name: str = "gemma3:1b", prompt: str = "What's the capital of France?"):
    """Run both methods and compare the results."""
    logger.info("-" * 50)
    logger.info(f"RUNNING COMPARISON TEST WITH MODEL: {model_name}")
    logger.info(f"PROMPT: {prompt}")
    logger.info("-" * 50)
    
    # First run the direct curl method
    logger.info("\nTESTING DIRECT CURL METHOD:")
    curl_result = test_direct_curl(model_name, prompt)
    
    # Then run our connector method
    logger.info("\nTESTING OLLAMA CONNECTOR:")
    connector = OllamaConnector()
    connector_result = connector.generate_completion_streaming(
        model_name=model_name,
        prompt=prompt
    )
    
    # Save connector result to a file
    with open("ollama_connector_comparison.json", "w") as f:
        json.dump(connector_result, f, indent=2)
    
    # Compare the results
    logger.info("\nCOMPARISON:")
    logger.info(f"Curl response length: {len(curl_result['response_text'])}")
    logger.info(f"Connector response length: {len(connector_result['response_text'])}")
    logger.info(f"Response texts match: {curl_result['response_text'] == connector_result['response_text']}")
    
    if 'token_count' in curl_result and 'token_count' in connector_result:
        logger.info(f"Curl token count: {curl_result['token_count']['total']}")
        logger.info(f"Connector token count: {connector_result['token_count']['total']}")
        logger.info(f"Token counts match: {curl_result['token_count']['total'] == connector_result['token_count']['total']}")
    
    logger.info(f"Curl latency: {curl_result['latency']:.2f} seconds")
    logger.info(f"Connector latency: {connector_result['latency']:.2f} seconds")
    
    return {
        "curl_result": curl_result,
        "connector_result": connector_result,
        "text_match": curl_result['response_text'] == connector_result['response_text'],
        "token_match": (curl_result.get('token_count', {}).get('total', 0) == 
                       connector_result.get('token_count', {}).get('total', 0))
    }

if __name__ == "__main__":
    # Get model name and prompt from command line arguments if provided
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma3:1b"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What's the capital of France?"
    
    # Run the comparison test
    comparison = test_comparison(model_name, prompt)
    
    # Summarize the results
    logger.info("\n" + "-" * 50)
    logger.info("TEST SUMMARY")
    logger.info("-" * 50)
    
    if comparison["text_match"]:
        logger.info("✅ PASS: Response texts match exactly")
    else:
        logger.info("❌ FAIL: Response texts differ")
    
    if comparison["token_match"]:
        logger.info("✅ PASS: Token counts match exactly")
    else:
        logger.info("❌ FAIL: Token counts differ")
        
    logger.info("-" * 50) 