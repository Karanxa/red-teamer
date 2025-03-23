"""
Model management utilities for RedTeamer framework.
This module provides functionality to dynamically fetch available models from different providers.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
import subprocess
import requests
from rich.console import Console

# Initialize console for rich output
console = Console()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model configurations - fallback if API requests fail
DEFAULT_MODELS = {
    "openai": {
        "gpt-4": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "OPENAI_API_KEY"
        },
        "gpt-3.5-turbo": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "OPENAI_API_KEY"
        }
    },
    "anthropic": {
        "claude-3-opus": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "claude-3-sonnet": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "claude-3-haiku": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        }
    },
    "gemini": {
        "gemini-pro": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "api_key_env": "GOOGLE_API_KEY"
        },
        "gemini-1.5-pro": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "api_key_env": "GOOGLE_API_KEY"
        },
        "gemini-1.5-flash": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "api_key_env": "GOOGLE_API_KEY"
        }
    },
    "ollama": {
        "llama3": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "mistral": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "phi3": {
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }
}

def get_openai_models() -> Dict[str, Dict[str, Any]]:
    """
    Fetch available models from OpenAI API.
    
    Returns:
        Dictionary of available models with their configurations
    """
    models = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.info("OpenAI API key not found in environment variables")
        return DEFAULT_MODELS.get("openai", {})
    
    try:
        # Check if openai package is installed
        import importlib.util
        openai_spec = importlib.util.find_spec("openai")
        
        if openai_spec is None:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            return DEFAULT_MODELS.get("openai", {})
        
        # Import the OpenAI module
        import openai
        from openai import OpenAI
        
        # Create client
        client = OpenAI(api_key=api_key)
        
        # Fetch available models
        response = client.models.list()
        
        for model in response.data:
            # Filter for GPT models only
            if "gpt" in model.id.lower():
                # Add model to the list
                models[model.id] = {
                    "temperature": 0.7,
                    "max_tokens": 1000,  # Default value
                    "api_key_env": "OPENAI_API_KEY"
                }
                
                # Use higher token limit for GPT-4 models
                if "gpt-4" in model.id.lower():
                    models[model.id]["max_tokens"] = 4096
        
        logger.info(f"Fetched {len(models)} models from OpenAI API")
        return models
        
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {str(e)}")
        return DEFAULT_MODELS.get("openai", {})

def get_anthropic_models() -> Dict[str, Dict[str, Any]]:
    """
    Get Anthropic models (currently using defaults as optional).
    
    Returns:
        Dictionary of available models with their configurations
    """
    # For now, just return the default Anthropic models
    # In the future, this could be expanded to fetch models from Anthropic API
    anthropic_models = DEFAULT_MODELS.get("anthropic", {})
    
    # Add a note that these are default models
    for model_config in anthropic_models.values():
        model_config["note"] = "API key required"
    
    return anthropic_models

def get_gemini_models() -> Dict[str, Dict[str, Any]]:
    """
    Fetch available models from Google Gemini API.
    
    Returns:
        Dictionary of available models with their configurations
    """
    models = {}
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.info("Google API key not found in environment variables")
        return DEFAULT_MODELS.get("gemini", {})
    
    try:
        # Check if google.generativeai package is installed
        import importlib.util
        genai_spec = importlib.util.find_spec("google.generativeai")
        
        if genai_spec is None:
            logger.warning("Google Generative AI package not installed. Install with: pip install google-generativeai")
            return DEFAULT_MODELS.get("gemini", {})
        
        # Import the Google Generative AI module
        import google.generativeai as genai
        
        # Configure with API key
        genai.configure(api_key=api_key)
        
        # Fetch available models
        model_list = genai.list_models()
        
        for model in model_list:
            # Filter for Gemini models only
            if "gemini" in model.name.lower():
                # Extract the model ID from the full name (e.g., "models/gemini-pro" -> "gemini-pro")
                model_id = model.name.split('/')[-1]
                
                # Add model to the list
                models[model_id] = {
                    "temperature": 0.7,
                    "max_tokens": 1024,  # Default value
                    "api_key_env": "GOOGLE_API_KEY"
                }
                
                # Use higher token limit for Pro models
                if "pro" in model_id.lower():
                    models[model_id]["max_tokens"] = 2048
        
        logger.info(f"Fetched {len(models)} models from Google Gemini API")
        return models
        
    except Exception as e:
        logger.error(f"Error fetching Google Gemini models: {str(e)}")
        return DEFAULT_MODELS.get("gemini", {})

def get_ollama_models() -> Dict[str, Dict[str, Any]]:
    """
    Fetch locally available models from Ollama server.
    
    Returns:
        Dictionary of available models with their configurations
    """
    models = {}
    
    try:
        # Try to connect to the Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the models
            for model_info in data.get("models", []):
                model_name = model_info.get("name")
                
                if model_name:
                    models[model_name] = {
                        "temperature": 0.7,
                        "max_tokens": 2048  # Default for most Ollama models
                    }
            
            logger.info(f"Fetched {len(models)} models from local Ollama server")
            return models
        else:
            logger.warning(f"Failed to fetch Ollama models: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to Ollama server: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
    
    # Alternative: Try using the Ollama CLI
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            # Parse the output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header line
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 1:
                        model_name = parts[0]
                        models[model_name] = {
                            "temperature": 0.7,
                            "max_tokens": 2048
                        }
            
            logger.info(f"Fetched {len(models)} models from Ollama CLI")
            return models
    except Exception as e:
        logger.error(f"Error using Ollama CLI: {str(e)}")
    
    # If all methods fail, return default models
    return DEFAULT_MODELS.get("ollama", {})

def get_all_available_models() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get all available models from all providers.
    
    Returns:
        Dictionary of providers and their available models
    """
    available_models = {}
    
    # OpenAI models
    openai_models = get_openai_models()
    if openai_models:
        available_models["openai"] = openai_models
    
    # Anthropic models
    anthropic_models = get_anthropic_models()
    if anthropic_models:
        available_models["anthropic"] = anthropic_models
    
    # Google Gemini models
    gemini_models = get_gemini_models()
    if gemini_models:
        available_models["gemini"] = gemini_models
    
    # Ollama models
    ollama_models = get_ollama_models()
    if ollama_models:
        available_models["ollama"] = ollama_models
    
    return available_models

def select_model_interactively() -> str:
    """
    Interactive CLI for selecting a model.
    
    Returns:
        Selected model in format "provider:model_id"
    """
    console.print("\n[bold]Fetching available models...[/bold]")
    
    # Get available models
    available_models = get_all_available_models()
    
    # Print available models by provider
    console.print("\n[bold]Available models:[/bold]")
    for provider, models in available_models.items():
        model_list = ", ".join(models.keys())
        console.print(f"{provider}: {model_list}")
    
    # Default provider and model
    default_provider = "openai"
    default_model = "gpt-4"
    
    if default_provider in available_models and available_models[default_provider]:
        default_model = next(iter(available_models[default_provider].keys()))
    
    # Get user selection
    selection = input(f"\nAdd model (format: provider:model, e.g. openai:gpt-4) ({default_provider}:{default_model}): ")
    
    if not selection:
        return f"{default_provider}:{default_model}"
    
    if ":" not in selection:
        console.print("[yellow]Invalid format. Using default model.[/yellow]")
        return f"{default_provider}:{default_model}"
    
    provider, model = selection.split(":", 1)
    
    # Validate selection
    if provider not in available_models:
        console.print(f"[yellow]Provider '{provider}' not available. Using default model.[/yellow]")
        return f"{default_provider}:{default_model}"
    
    if model not in available_models[provider]:
        console.print(f"[yellow]Model '{model}' not available for provider '{provider}'. Using default model.[/yellow]")
        return f"{default_provider}:{default_model}"
    
    # Handle API key requirement
    model_config = available_models[provider][model]
    api_key_env = model_config.get("api_key_env")
    
    if api_key_env and api_key_env not in os.environ:
        console.print(f"[yellow]Warning: API key not found in environment variable {api_key_env}[/yellow]")
        api_key = input(f"Enter your {provider.upper()} API key (leave empty to skip): ")
        
        if api_key.strip():
            # Set environment variable for this session
            os.environ[api_key_env] = api_key.strip()
            console.print(f"[green]API key for {provider} has been set for this session.[/green]")
        else:
            console.print(f"[yellow]No API key provided. Model may not work correctly.[/yellow]")
    
    return f"{provider}:{model}"

if __name__ == "__main__":
    # For testing/debugging
    models = get_all_available_models()
    print(json.dumps(models, indent=2))
    
    selected = select_model_interactively()
    print(f"Selected model: {selected}") 