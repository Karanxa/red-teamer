"""
API Key Manager for the Red Teamer tool.

This module provides centralized API key management for various model providers.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

# Configure logging
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for various model providers."""
    
    # Constants
    CONFIG_DIR = ".redteamer"
    KEYS_FILE = "api_keys.json"
    
    # Map provider names to environment variable names
    PROVIDER_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",  # Alias for gemini
        "huggingface": "HUGGINGFACE_API_KEY"
    }
    
    # Map provider names to their display names
    PROVIDER_DISPLAY_NAMES = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Google Gemini",
        "google": "Google Gemini",
        "huggingface": "Hugging Face"
    }
    
    def __init__(self):
        """Initialize the API Key Manager."""
        self.config_path = self._get_config_path()
        self.keys_file_path = self.config_path / self.KEYS_FILE
        self._keys = self._load_keys()
    
    def _get_config_path(self) -> Path:
        """Get the configuration directory path."""
        # Use the user's home directory for storing keys
        home_dir = Path.home()
        config_dir = home_dir / self.CONFIG_DIR
        
        # Create the directory if it doesn't exist
        if not config_dir.exists():
            config_dir.mkdir(exist_ok=True)
            # Secure the directory permissions (only on Unix-like systems)
            if os.name == "posix":
                os.chmod(config_dir, 0o700)  # Only the user can access this directory
        
        return config_dir
    
    def _load_keys(self) -> Dict[str, str]:
        """Load API keys from the keys file."""
        if not self.keys_file_path.exists():
            # Create an empty keys file if it doesn't exist
            self._save_keys({})
            return {}
        
        try:
            with open(self.keys_file_path, "r") as f:
                keys = json.load(f)
            # Ensure the correct structure
            if not isinstance(keys, dict):
                logger.warning("Keys file has invalid format. Resetting to empty.")
                keys = {}
                self._save_keys(keys)
            return keys
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return {}
    
    def _save_keys(self, keys: Dict[str, str]) -> None:
        """Save API keys to the keys file."""
        try:
            with open(self.keys_file_path, "w") as f:
                json.dump(keys, f, indent=2)
            
            # Secure the file permissions (only on Unix-like systems)
            if os.name == "posix":
                os.chmod(self.keys_file_path, 0o600)  # Only the user can read/write this file
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def set_key(self, provider: str, key: str) -> bool:
        """
        Set an API key for a provider.
        
        Args:
            provider: Provider name (e.g., openai, anthropic, gemini, huggingface)
            key: API key value
            
        Returns:
            bool: True if successful, False otherwise
        """
        provider = provider.lower()
        
        # Validate provider
        if provider not in self.PROVIDER_ENV_VARS:
            logger.error(f"Unknown provider: {provider}")
            return False
        
        # Store the key
        self._keys[provider] = key
        self._save_keys(self._keys)
        
        # Also set the environment variable for immediate use
        os.environ[self.PROVIDER_ENV_VARS[provider]] = key
        
        return True
    
    def get_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a provider.
        
        Args:
            provider: Provider name (e.g., openai, anthropic, gemini, huggingface)
            
        Returns:
            Optional[str]: API key if found, None otherwise
        """
        provider = provider.lower()
        
        # Check if provider is valid
        if provider not in self.PROVIDER_ENV_VARS:
            logger.error(f"Unknown provider: {provider}")
            return None
        
        # First check environment variable
        env_var = self.PROVIDER_ENV_VARS[provider]
        env_key = os.environ.get(env_var)
        
        if env_key:
            # If found in environment, update stored key to ensure consistency
            if env_key != self._keys.get(provider):
                self._keys[provider] = env_key
                self._save_keys(self._keys)
            return env_key
        
        # If not in environment, check stored keys
        key = self._keys.get(provider)
        if key:
            # Update environment variable for subsequent calls
            os.environ[env_var] = key
        
        return key
    
    def delete_key(self, provider: str) -> bool:
        """
        Delete an API key for a provider.
        
        Args:
            provider: Provider name (e.g., openai, anthropic, gemini, huggingface)
            
        Returns:
            bool: True if the key was deleted, False otherwise
        """
        provider = provider.lower()
        
        # Check if provider is valid
        if provider not in self.PROVIDER_ENV_VARS:
            logger.error(f"Unknown provider: {provider}")
            return False
        
        # Remove from stored keys
        if provider in self._keys:
            del self._keys[provider]
            self._save_keys(self._keys)
        
        # Remove from environment if present
        env_var = self.PROVIDER_ENV_VARS[provider]
        if env_var in os.environ:
            del os.environ[env_var]
        
        return True
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """
        Get a list of all supported providers with their status.
        
        Returns:
            List of dictionaries containing provider information
        """
        providers = []
        
        for provider, env_var in self.PROVIDER_ENV_VARS.items():
            # Skip aliases
            if provider == "google":  # Skip "google" alias for "gemini"
                continue
                
            key = self.get_key(provider)
            display_name = self.PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())
            
            providers.append({
                "provider": provider,
                "display_name": display_name,
                "env_var": env_var,
                "has_key": key is not None,
                "key_preview": f"{key[:4]}...{key[-4:]}" if key else None
            })
        
        return providers
    
    def ensure_key_env_var(self, provider: str) -> bool:
        """
        Ensure that an API key for a provider is available in the environment.
        This doesn't return the key itself but makes sure it's set in the environment.
        
        Args:
            provider: Provider name
            
        Returns:
            bool: True if the key is available, False otherwise
        """
        key = self.get_key(provider)
        if not key:
            return False
            
        # Key is already set in environment by get_key()
        return True


# Create a singleton instance
_api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """
    Get the API Key Manager singleton instance.
    
    Returns:
        APIKeyManager: The API Key Manager instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager 