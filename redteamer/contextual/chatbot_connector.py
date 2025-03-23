"""
Chatbot Connector for interfacing with external chatbots via curl commands.

This module provides functionality to interact with external chatbots
using customizable curl commands.
"""

import os
import time
import json
import logging
import subprocess
import shlex
import re
from typing import Dict, List, Any, Optional, Union

class ChatbotConnector:
    """
    Connector for interacting with external chatbots via curl commands.
    
    This class handles sending prompts to chatbots and receiving responses
    using customizable curl commands.
    """
    
    def __init__(self, curl_template: str, verbose: bool = False):
        """
        Initialize the chatbot connector.
        
        Args:
            curl_template: Curl command template with {prompt} placeholder
            verbose: Whether to print verbose output during interactions
        """
        self.logger = logging.getLogger(__name__)
        self.curl_template = curl_template
        self.verbose = verbose
        
        # Validate the curl template
        if "{prompt}" not in self.curl_template:
            self.logger.error("Invalid curl template: Must contain {prompt} placeholder")
            raise ValueError("Invalid curl template: Must contain {prompt} placeholder")
    
    def send_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a prompt to the chatbot and return the response.
        
        Args:
            prompt: The prompt to send to the chatbot
            system_prompt: Optional system prompt (if supported in the curl template)
            
        Returns:
            Dictionary with the chatbot response and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the curl command with the prompt and system prompt
            filled_command = self.curl_template
            
            # Escape the prompt for shell
            escaped_prompt = prompt.replace('"', '\\"').replace('$', '\\$')
            
            # Replace prompt placeholder
            filled_command = filled_command.replace("{prompt}", escaped_prompt)
            
            # Replace system prompt placeholder if it exists
            if "{system_prompt}" in filled_command and system_prompt:
                escaped_system = system_prompt.replace('"', '\\"').replace('$', '\\$')
                filled_command = filled_command.replace("{system_prompt}", escaped_system)
            elif "{system_prompt}" in filled_command:
                # Replace with empty string if no system prompt provided
                filled_command = filled_command.replace("{system_prompt}", "")
            
            if self.verbose:
                self.logger.debug(f"Executing curl command: {filled_command}")
            
            # Execute the curl command
            process = subprocess.Popen(
                filled_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            
            stdout, stderr = process.communicate()
            
            # Check if there was an error
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                self.logger.error(f"Curl command failed: {error_msg}")
                raise Exception(f"Curl command failed: {error_msg}")
            
            # Parse the response
            response_text = stdout.decode('utf-8')
            
            # Try to parse as JSON if it looks like JSON
            response_data = {}
            try:
                if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                    response_data = json.loads(response_text)
                    
                    # Try to extract the text from common response formats
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        # OpenAI-like format
                        if 'message' in response_data['choices'][0]:
                            response_text = response_data['choices'][0]['message'].get('content', '')
                        elif 'text' in response_data['choices'][0]:
                            response_text = response_data['choices'][0]['text']
                    elif 'content' in response_data:
                        # Anthropic-like format
                        response_text = response_data['content']
                    elif 'text' in response_data:
                        # Simple format
                        response_text = response_data['text']
                    elif 'response' in response_data:
                        # Another common format
                        response_text = response_data['response']
                    elif 'message' in response_data:
                        # Yet another format
                        response_text = response_data['message']
            except json.JSONDecodeError:
                # Not JSON, use raw response
                pass
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())
            
            return {
                'prompt': prompt,
                'response_text': response_text,
                'tokens': prompt_tokens + completion_tokens,
                'latency': elapsed_time,
                'response_data': response_data,
                'token_count': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': prompt_tokens + completion_tokens
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error sending prompt to chatbot: {e}")
            # Calculate elapsed time even for errors
            elapsed_time = time.time() - start_time
            
            return {
                'prompt': prompt,
                'response_text': f"Error: {str(e)}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': str(e)
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the chatbot with a harmless prompt.
        
        Returns:
            Dictionary with the test results
        """
        test_prompt = "Hello, how are you today?"
        
        try:
            if self.verbose:
                self.logger.info("Testing chatbot connection with a benign prompt")
            
            result = self.send_prompt(test_prompt)
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "message": "Failed to connect to chatbot"
                }
            
            return {
                "success": True,
                "response": result["response_text"],
                "latency": result["latency"],
                "message": "Successfully connected to chatbot"
            }
            
        except Exception as e:
            self.logger.error(f"Error testing chatbot connection: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to chatbot"
            } 