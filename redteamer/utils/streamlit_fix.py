"""
Fix for PyTorch-Streamlit compatibility issues.

This module provides a solution for the issues that occur when using PyTorch
with Streamlit, specifically the error with torch.classes.__path__._path.

Usage:
    1. Set the environment variable REDTEAMER_FIX_STREAMLIT=1 before running Streamlit
    2. Import this module at the beginning of your Streamlit app:
       `import redteamer.utils.streamlit_fix`
"""

import os
import sys
import logging
import asyncio
from typing import List, Optional, Any, Dict, Set

# Configure logging
logger = logging.getLogger(__name__)

# List of modules to exclude from Streamlit's file watcher
EXCLUDED_MODULES = {"torch", "transformers", "accelerate", "bitsandbytes", "optimum"}

def apply_torch_streamlit_fix():
    """
    Apply fixes to make PyTorch work with Streamlit.
    
    This function will:
    1. Monkey patch the Streamlit file watcher to ignore problematic modules
    2. Set up exception handlers for common PyTorch-Streamlit issues
    3. Add a fix for the 'no running event loop' error
    """
    # Check if the environment variable is set to enable the fix
    if os.environ.get("REDTEAMER_FIX_STREAMLIT", "0") != "1":
        return False
    
    try:
        # Fix for the 'no running event loop' error
        _patch_asyncio_event_loop()
        
        # Only apply if streamlit is installed
        try:
            import streamlit
            HAS_STREAMLIT = True
        except ImportError:
            HAS_STREAMLIT = False
            return False
        
        # Patch the local_sources_watcher module
        if HAS_STREAMLIT and "streamlit.watcher.local_sources_watcher" in sys.modules:
            module = sys.modules["streamlit.watcher.local_sources_watcher"]
            _patch_local_sources_watcher(module)
            logger.info("Applied Streamlit file watcher patch for PyTorch compatibility")
            return True
        
        # If streamlit.watcher.local_sources_watcher hasn't been imported yet,
        # we need to set up an import hook to patch it when it's imported
        class StreamlitPatchFinder:
            """Import hook to patch Streamlit modules when they're imported."""
            
            def __init__(self):
                self.patched_modules: Set[str] = set()
            
            def find_spec(self, fullname, path, target=None):
                # We're not actually finding modules, just intercepting them
                return None
            
            def exec_module(self, module):
                # We're not executing modules
                pass
            
            def load_module(self, fullname):
                # Let the default machinery load the module
                module = importlib.import_module(fullname)
                
                # Patch the module if it's the one we're looking for
                if fullname == "streamlit.watcher.local_sources_watcher" and fullname not in self.patched_modules:
                    _patch_local_sources_watcher(module)
                    self.patched_modules.add(fullname)
                    logger.info(f"Patched {fullname} for PyTorch compatibility")
                
                return module
        
        # Install the import hook
        import importlib
        sys.meta_path.insert(0, StreamlitPatchFinder())
        logger.info("Installed import hook for Streamlit file watcher patch")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply Streamlit/PyTorch fix: {e}")
        return False

def _patch_asyncio_event_loop():
    """
    Patch the asyncio module to handle 'no running event loop' errors.
    
    This works by ensuring there's always a running event loop when requested.
    """
    try:
        # Store the original get_running_loop function
        original_get_running_loop = asyncio.get_running_loop
        
        def patched_get_running_loop():
            """Patched version that creates an event loop if none exists."""
            try:
                return original_get_running_loop()
            except RuntimeError:
                # No running event loop, create one
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop
                except Exception as e:
                    logger.error(f"Failed to create new event loop: {e}")
                    raise
        
        # Replace the original function with our patched version
        asyncio.get_running_loop = patched_get_running_loop
        logger.info("Patched asyncio.get_running_loop to handle 'no running event loop' errors")
        return True
    except Exception as e:
        logger.error(f"Failed to patch asyncio.get_running_loop: {e}")
        return False

def _patch_local_sources_watcher(module):
    """
    Patch Streamlit's LocalSourcesWatcher to ignore problematic modules.
    
    Args:
        module: The streamlit.watcher.local_sources_watcher module
    """
    if not hasattr(module, "extract_paths"):
        logger.warning("extract_paths function not found in local_sources_watcher")
        return False
    
    original_extract_paths = module.extract_paths
    
    def patched_extract_paths(module_object):
        """Patched version that handles problematic modules."""
        # Skip modules that cause problems
        module_name = getattr(module_object, "__name__", "")
        for excluded_prefix in EXCLUDED_MODULES:
            if module_name.startswith(f"{excluded_prefix}.") or module_name == excluded_prefix:
                return []
        
        # Use the original function for other modules
        try:
            return original_extract_paths(module_object)
        except Exception as e:
            # If there's an error, log it and return an empty list
            logger.debug(f"Error extracting paths from {module_name}: {e}")
            return []
    
    # Replace the original function with our patched version
    module.extract_paths = patched_extract_paths
    return True

# Apply the fix automatically when this module is imported
fix_applied = apply_torch_streamlit_fix()
if fix_applied:
    logger.info("Successfully applied PyTorch-Streamlit compatibility fix")

# Function to safely import streamlit
def safe_import_streamlit():
    """
    Safely import streamlit with the PyTorch fix applied.
    
    Returns:
        The streamlit module
    """
    # Make sure our fix is enabled
    os.environ["REDTEAMER_FIX_STREAMLIT"] = "1"
    
    # Apply the fix if not already applied
    if not fix_applied:
        apply_torch_streamlit_fix()
    
    # Import streamlit
    try:
        import streamlit as st
        return st
    except Exception as e:
        logger.error(f"Failed to import streamlit: {e}")
        raise 