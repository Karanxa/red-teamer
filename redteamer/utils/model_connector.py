"""
Model connector for interacting with various LLM APIs.
"""

import os
import time
import json
import logging
import importlib.util
import subprocess
import shlex
import re
import requests
from typing import Dict, List, Optional, Union, Any, Callable

# Import API clients
import openai
import anthropic
import google.generativeai as genai

# Add custom model connector for curl commands
class CustomModelConnector:
    """Connector for custom models via curl commands."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_completion(self, curl_command: str, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a completion from a custom model using a curl command.
        
        Args:
            curl_command: Curl command template with {prompt} and optional {system_prompt} placeholders
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
        
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the curl command with the prompt and system prompt
            filled_command = curl_command
            
            # Escape the prompt and system prompt for shell
            escaped_prompt = prompt.replace('"', '\\"')
            
            # Replace prompt placeholder
            filled_command = filled_command.replace("{prompt}", escaped_prompt)
            
            # Replace system prompt placeholder if it exists
            if "{system_prompt}" in filled_command and system_prompt:
                escaped_system_prompt = system_prompt.replace('"', '\\"')
                filled_command = filled_command.replace("{system_prompt}", escaped_system_prompt)
            elif "{system_prompt}" in filled_command:
                # Replace with empty string if no system prompt provided
                filled_command = filled_command.replace("{system_prompt}", "")
            
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
            except json.JSONDecodeError:
                # Not JSON, use raw response
                pass
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())
            
            return {
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
            self.logger.error(f"Error generating completion with custom model: {e}")
            # Calculate elapsed time even for errors
            elapsed_time = time.time() - start_time
            
            return {
                'response_text': f"Error: {str(e)}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': str(e)
            }

class OllamaConnector:
    """Connector for Ollama models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "http://localhost:11434"  # Default Ollama API address
    
    def set_base_url(self, base_url: str):
        """Set a custom base URL for the Ollama API."""
        self.base_url = base_url
    
    def generate_completion_streaming(self, model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                           temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a completion from an Ollama model using the streaming API.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        api_url = f"{self.base_url}/api/generate"
        
        try:
            # Prepare the request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            self.logger.debug(f"Sending streaming request to Ollama API: {payload}")
            
            # Make the API request with stream=True to get the response in chunks
            response = requests.post(api_url, json=payload, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Variables to store the accumulated response
            accumulated_response = ""
            final_response_data = None
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Parse the JSON chunk
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        
                        # Accumulate the response text
                        if 'response' in chunk:
                            accumulated_response += chunk['response']
                        
                        # If this is the final chunk (done=True), save the metadata
                        if chunk.get('done', False):
                            final_response_data = chunk
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing JSON chunk: {e}")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Extract or estimate token counts from the final response data
            prompt_tokens = (final_response_data.get("prompt_eval_count", 0) 
                            if final_response_data else len(prompt.split()))
            completion_tokens = (final_response_data.get("eval_count", 0) 
                               if final_response_data else len(accumulated_response.split()))
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                'response_text': accumulated_response,
                'tokens': total_tokens,
                'latency': elapsed_time,
                'response_data': final_response_data,
                'token_count': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': total_tokens
                }
            }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error connecting to Ollama API: {e}")
            elapsed_time = time.time() - start_time
            
            # Check if it's a connection error - provide helpful message for common issues
            if isinstance(e, requests.exceptions.ConnectionError):
                error_msg = (f"Could not connect to Ollama at {self.base_url}. "
                            f"Make sure Ollama is running and accessible. "
                            f"You can start Ollama by running 'ollama serve' in your terminal.")
            else:
                error_msg = str(e)
            
            return {
                'response_text': f"Error: {error_msg}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': error_msg
            }
        except Exception as e:
            self.logger.error(f"Error generating completion with Ollama: {e}")
            elapsed_time = time.time() - start_time
            
            return {
                'response_text': f"Error: {str(e)}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': str(e)
            }
    
    def generate_completion(self, model_name: str, prompt: str, system_prompt: Optional[str] = None, 
                           temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a completion from an Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        api_url = f"{self.base_url}/api/generate"
        
        try:
            # Prepare the request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "raw": True  # Get the raw completion
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            self.logger.debug(f"Sending request to Ollama API: {payload}")
            
            # Try first with the streaming method as it's more reliable
            try:
                return self.generate_completion_streaming(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as stream_error:
                self.logger.warning(f"Streaming request failed, falling back to non-streaming: {stream_error}")
            
            # Make the API request
            response = requests.post(api_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the response
            response_data = response.json()
            
            # Extract the response text
            response_text = response_data.get("response", "")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Extract or estimate token counts
            prompt_tokens = response_data.get("prompt_eval_count", len(prompt.split()))
            completion_tokens = response_data.get("eval_count", len(response_text.split()))
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                'response_text': response_text,
                'tokens': total_tokens,
                'latency': elapsed_time,
                'response_data': response_data,
                'token_count': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': total_tokens
                }
            }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error connecting to Ollama API: {e}")
            elapsed_time = time.time() - start_time
            
            # Check if it's a connection error - provide helpful message for common issues
            if isinstance(e, requests.exceptions.ConnectionError):
                error_msg = (f"Could not connect to Ollama at {self.base_url}. "
                            f"Make sure Ollama is running and accessible. "
                            f"You can start Ollama by running 'ollama serve' in your terminal.")
            else:
                error_msg = str(e)
            
            return {
                'response_text': f"Error: {error_msg}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': error_msg
            }
        except Exception as e:
            self.logger.error(f"Error generating completion with Ollama: {e}")
            elapsed_time = time.time() - start_time
            
            return {
                'response_text': f"Error: {str(e)}",
                'tokens': 0,
                'latency': elapsed_time,
                'error': str(e)
            }

class ModelConnector:
    """
    Connector for interacting with various LLM providers.
    
    This class handles the communication with different LLM APIs,
    providing a unified interface for the benchmark engine.
    """
    
    def __init__(self):
        """Initialize the ModelConnector."""
        self.logger = logging.getLogger(__name__)
        self._clients = {}
        self.custom_connector = CustomModelConnector()
        self.ollama_connector = OllamaConnector()
    
    def _get_openai_client(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Get an OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
            api_base: OpenAI API base URL. If None, uses default.
            
        Returns:
            OpenAI client.
        """
        try:
            # Check if OpenAI package is installed
            if importlib.util.find_spec("openai") is None:
                self.logger.error("OpenAI package not installed. Install with 'pip install openai'")
                raise ImportError("OpenAI package not installed")
            
            # Configure API key
            client_params = {}
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
            if not openai.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
            
            # Configure API base if provided
            if api_base:
                client_params["base_url"] = api_base
            
            # Create client
            client = openai.OpenAI(**client_params)
            return client
        
        except (ImportError, ValueError) as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def _get_anthropic_client(self, api_key: Optional[str] = None):
        """
        Get an Anthropic client.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable.
            
        Returns:
            Anthropic client.
        """
        try:
            # Check if Anthropic package is installed
            if importlib.util.find_spec("anthropic") is None:
                self.logger.error("Anthropic package not installed. Install with 'pip install anthropic'")
                raise ImportError("Anthropic package not installed")
            
            # Configure API key
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            
            if not api_key:
                raise ValueError("Anthropic API key not provided and not found in environment variables")
            
            # Create client
            client = anthropic.Anthropic(api_key=api_key)
            return client
        
        except (ImportError, ValueError) as e:
            self.logger.error(f"Error initializing Anthropic client: {e}")
            raise
    
    def _get_gemini_client(self, api_key: Optional[str] = None):
        """
        Get a Google Gemini client.
        
        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY environment variable.
            
        Returns:
            Configured Google Gemini client
        """
        try:
            # Check if Google Generative AI package is installed
            if importlib.util.find_spec("google.generativeai") is None:
                self.logger.error("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
                raise ImportError("Google Generative AI package not installed")
            
            # Configure API key
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError("Google API key not provided and not found in environment variables")
            
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # Store a reference to the module as our "client"
            self._clients["gemini"] = genai
            return genai
        
        except (ImportError, ValueError) as e:
            self.logger.error(f"Error initializing Google Gemini client: {e}")
            raise
    
    def _configure_ollama_connector(self, model_config: Dict[str, Any]):
        """
        Configure the Ollama connector with custom settings.
        
        Args:
            model_config: Ollama model configuration
        """
        # Set custom base URL if provided
        if "api_base" in model_config:
            self.ollama_connector.set_base_url(model_config["api_base"])
        elif "OLLAMA_API_BASE" in os.environ:
            self.ollama_connector.set_base_url(os.environ["OLLAMA_API_BASE"])
    
    def get_client(self, provider: str, model_config: Dict):
        """
        Get a client for the specified provider.
        
        Args:
            provider: Provider name (openai, anthropic, gemini, ollama)
            model_config: Model configuration
            
        Returns:
            Client for the specified provider
        """
        provider = provider.lower()
        
        if provider not in self._clients:
            # Initialize client based on provider
            if provider == "openai":
                api_key = None
                if "api_key_env" in model_config:
                    api_key = os.environ.get(model_config["api_key_env"])
                
                api_base = model_config.get("api_base_url")
                
                self._clients[provider] = self._get_openai_client(api_key, api_base)
            
            elif provider == "anthropic":
                api_key = None
                if "api_key_env" in model_config:
                    api_key = os.environ.get(model_config["api_key_env"])
                
                self._clients[provider] = self._get_anthropic_client(api_key)
            
            elif provider in ["google", "gemini"]:
                api_key = None
                if "api_key_env" in model_config:
                    api_key = os.environ.get(model_config["api_key_env"])
                
                self._clients[provider] = self._get_gemini_client(api_key)
            
            elif provider == "ollama":
                # Ollama doesn't need a client, but we'll configure the connector
                self._configure_ollama_connector(model_config)
                self._clients[provider] = self.ollama_connector
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        return self._clients[provider]
    
    def generate_completion(self, model_config: Dict, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Generate a completion from a model.
        
        Args:
            model_config: Model configuration.
            prompt: User prompt.
            system_prompt: System prompt (for models that support it).
            
        Returns:
            Dictionary with the completion result.
        """
        start_time = time.time()
        provider = model_config["provider"].lower()
        model_id = model_config["model_id"]
        params = model_config.get("parameters", {})
        api_key_env = model_config.get("api_key_env")
        
        self.logger.debug(f"Generating completion with {provider}/{model_id}")
        
        try:
            if provider == "custom":
                # Handle custom model via curl
                curl_command = model_config.get("curl_command", "")
                if not curl_command:
                    raise ValueError("Missing curl_command in model configuration")
                
                return self.custom_connector.generate_completion(curl_command, prompt, system_prompt)
                
            elif provider == "ollama":
                # Handle Ollama model
                self._configure_ollama_connector(model_config)
                temperature = params.get("temperature", 0.7)
                max_tokens = params.get("max_tokens", 1000)
                
                result = self.ollama_connector.generate_completion(
                    model_name=model_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Add provider and model_id to the result
                result["provider"] = provider
                result["model_id"] = model_id
                result["success"] = "error" not in result
                result["latency_ms"] = int(result.get("latency", 0) * 1000)
                
                return result
                
            elif provider == "openai":
                return self._generate_openai_completion(model_id, prompt, system_prompt, params)
            
            elif provider == "anthropic":
                return self._generate_anthropic_completion(model_id, prompt, system_prompt, params)
            
            elif provider in ["google", "gemini"]:
                return self._generate_gemini_completion(model_id, prompt, system_prompt, params)
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            self.logger.error(f"Error generating completion: {e}")
            elapsed_time = time.time() - start_time
            
            return {
                "provider": provider,
                "model_id": model_id,
                "response_text": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "latency_ms": round(elapsed_time * 1000)
            }
    
    def _generate_openai_completion(self, model_id: str, prompt: str, system_prompt: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using OpenAI's API.
        
        Args:
            model_id: Model ID
            prompt: User prompt
            system_prompt: System prompt
            parameters: Additional parameters for the API call
            
        Returns:
            Dictionary with the completion result
        """
        client = self.get_client("openai", parameters)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Extract parameters
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 1000)
        
        # Make API call
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Extract response text
        response_text = completion.choices[0].message.content
        
        # Extract token counts
        token_count = {
            "prompt": completion.usage.prompt_tokens,
            "completion": completion.usage.completion_tokens,
            "total": completion.usage.total_tokens
        }
        
        return {
            "provider": "openai",
            "model_id": model_id,
            "response_text": response_text,
            "success": True,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "extra_metrics": {}
        }
    
    def _generate_anthropic_completion(self, model_id: str, prompt: str, system_prompt: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using Anthropic's API.
        
        Args:
            model_id: Model ID
            prompt: User prompt
            system_prompt: System prompt
            parameters: Additional parameters for the API call
            
        Returns:
            Dictionary with the completion result
        """
        client = self.get_client("anthropic", parameters)
        
        # Extract parameters
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 1000)
        
        # Make API call
        start_time = time.time()
        
        message_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
        
        completion = client.messages.create(**message_params)
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Extract response text
        response_text = completion.content[0].text
        
        # Extract token counts
        token_count = {
            "prompt": getattr(completion.usage, "input_tokens", 0),
            "completion": getattr(completion.usage, "output_tokens", 0),
            "total": getattr(completion.usage, "input_tokens", 0) + getattr(completion.usage, "output_tokens", 0)
        }
        
        return {
            "provider": "anthropic",
            "model_id": model_id,
            "response_text": response_text,
            "success": True,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "extra_metrics": {}
        }
    
    def _generate_gemini_completion(self, model_id: str, prompt: str, system_prompt: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using Google's Gemini API.
        
        Args:
            model_id: Model ID
            prompt: User prompt
            system_prompt: System prompt
            parameters: Additional parameters for the API call
            
        Returns:
            Dictionary with the completion result
        """
        client = self.get_client("gemini", parameters)
        
        # Extract parameters
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 1000)
        
        # Configure the model
        model = client.GenerativeModel(model_name=model_id)
        
        # Prepare the chat session if system prompt is provided
        if system_prompt:
            chat = model.start_chat(history=[
                {"role": "user", "parts": [system_prompt]},
                {"role": "model", "parts": ["I'll help you as requested."]}
            ])
            
            # Make API call
            start_time = time.time()
            response = chat.send_message(
                prompt,
                generation_config=client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
        else:
            # Make API call without system prompt
            start_time = time.time()
            response = model.generate_content(
                prompt,
                generation_config=client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Extract response text
        response_text = response.text
        
        # Gemini doesn't provide token counts in the API response
        # We'll estimate based on characters (roughly 4 chars per token)
        char_count = len(prompt) + len(response_text)
        estimated_tokens = char_count // 4
        
        # Estimate prompt vs completion split (30/70 split as estimation)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        
        token_count = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "estimated": True  # Flag to indicate these are estimated
        }
        
        return {
            "provider": "gemini",
            "model_id": model_id,
            "response_text": response_text,
            "success": True,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "extra_metrics": {}
        } 