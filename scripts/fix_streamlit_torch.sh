#!/bin/bash
# Helper script to run Streamlit applications with PyTorch compatibility fix
# Usage: ./scripts/fix_streamlit_torch.sh [command]
# Examples:
#   ./scripts/fix_streamlit_torch.sh python -m redteamer.conversational_redteam_launcher
#   ./scripts/fix_streamlit_torch.sh python -m redteamer.static_scan_launcher
#   ./scripts/fix_streamlit_torch.sh python -m redteamer.results_viewer_launcher

# Set the environment variable to enable the fix
export REDTEAMER_FIX_STREAMLIT=1

# Check if command is provided
if [ $# -eq 0 ]; then
    echo "Error: No command provided."
    echo "Usage: $0 [command]"
    echo "Examples:"
    echo "  $0 python -m redteamer.conversational_redteam_launcher"
    echo "  $0 python -m redteamer.static_scan_launcher"
    echo "  $0 python -m redteamer.results_viewer_launcher"
    exit 1
fi

# Execute the provided command
echo "Running command with Streamlit-PyTorch compatibility fix enabled..."
"$@" 