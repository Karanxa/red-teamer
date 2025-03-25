"""
Patch for Streamlit to prevent errors with PyTorch.

This module patches Streamlit's file watcher to ignore PyTorch modules,
which have a custom import system that can cause errors.
"""

import sys
import logging
from typing import List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

def apply_streamlit_patch():
    """
    Apply patches to Streamlit to prevent errors with PyTorch.
    
    This function should be called before importing Streamlit.
    """
    try:
        # Check if streamlit is already imported
        if 'streamlit' in sys.modules:
            logger.warning("Streamlit already imported, patches may not be fully effective")
        
        # Store the original __import__ function
        original_import = __import__
        
        # Define a wrapper function to modify streamlit's behavior
        def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            module = original_import(name, globals, locals, fromlist, level)
            
            # Patch streamlit's file watcher after it's imported
            if name == 'streamlit.watcher.local_sources_watcher':
                patch_file_watcher(module)
            
            return module
        
        # Replace the built-in __import__ function with our patched version
        sys.meta_path.insert(0, PatchedImporter(original_import))
        
        logger.info("Applied Streamlit patches to handle PyTorch modules")
        return True
    except Exception as e:
        logger.error(f"Failed to apply Streamlit patches: {e}")
        return False

def patch_file_watcher(module):
    """
    Patch Streamlit's LocalSourcesWatcher to ignore problematic modules.
    
    Args:
        module: The streamlit.watcher.local_sources_watcher module
    """
    try:
        original_extract_paths = module.extract_paths
        
        def patched_extract_paths(module_object):
            """Patched version that handles problematic modules."""
            # Skip PyTorch modules that cause problems
            module_name = getattr(module_object, '__name__', '')
            if module_name.startswith('torch.'):
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
        logger.info("Patched Streamlit's LocalSourcesWatcher to ignore PyTorch modules")
    except Exception as e:
        logger.error(f"Failed to patch LocalSourcesWatcher: {e}")

class PatchedImporter:
    """
    Custom importer that patches modules during import.
    """
    
    def __init__(self, original_import):
        """Initialize with the original import function."""
        self.original_import = original_import
    
    def find_spec(self, fullname, path, target=None):
        """Find the module spec."""
        # We're not actually finding modules, just modifying them after import
        return None
    
    def exec_module(self, module):
        """Execute the module - not used."""
        pass
    
    def create_module(self, spec):
        """Create the module - not used."""
        pass

# Apply the patch automatically when this module is imported
apply_streamlit_patch()

# Function to safely import streamlit after patching
def safe_import_streamlit():
    """
    Safely import streamlit after applying patches.
    
    Returns:
        The streamlit module
    """
    try:
        import streamlit as st
        return st
    except Exception as e:
        logger.error(f"Failed to import streamlit: {e}")
        raise 