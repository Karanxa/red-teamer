"""
Wrapper module for the interactive menu to avoid API key checks at startup.

This module provides a simple wrapper around the interactive menu
to prevent checking API keys immediately on import.
"""

def run_menu():
    """Run the interactive menu without checking API keys at startup."""
    # Import the run function here to prevent API key checks at startup
    from redteamer.interactive_menu import run
    run()

if __name__ == "__main__":
    run_menu() 