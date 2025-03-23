#!/bin/bash
# Run Red Teaming Framework helper script
# Usage: ./run_red_teamer.sh [command] [arguments]

# Set Python path - change if needed
PYTHON="python3"

# Check if Python is installed
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: Python is not installed or not in the PATH."
    echo "Please install Python 3.8 or later."
    exit 1
fi

# Check if redteamer package is installed
if ! $PYTHON -c "import redteamer" &> /dev/null; then
    echo "Installing Red Teaming Framework..."
    $PYTHON -m pip install -e .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Red Teaming Framework."
        exit 1
    fi
    echo "Red Teaming Framework installed successfully."
fi

# Check environment variables for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable is not set."
    echo "Set it with: export OPENAI_API_KEY=your_api_key"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY environment variable is not set."
    echo "Set it with: export ANTHROPIC_API_KEY=your_api_key"
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Warning: GOOGLE_API_KEY environment variable is not set."
    echo "Set it with: export GOOGLE_API_KEY=your_api_key"
fi

# Make directories if they don't exist
mkdir -p results datasets reports

# Run the command
if [ $# -eq 0 ]; then
    echo "Red Teaming Framework - Interactive CLI"
    echo "--------------------------------------"
    echo
    echo "First time using the framework? Start with:"
    echo "  ./run_red_teamer.sh quickstart"
    echo
    echo "This will guide you through:"
    echo "- Setting up your API keys"
    echo "- Running a simple red team evaluation"
    echo "- Generating a report"
    echo
    echo "Other example commands:"
    echo "  ./run_red_teamer.sh test      - Test a model interactively"
    echo "  ./run_red_teamer.sh run       - Run a red team evaluation with interactive configuration" 
    echo "  ./run_red_teamer.sh dataset create      - Create a dataset interactively"
    echo "  ./run_red_teamer.sh dataset add-vector  - Add vectors to a dataset interactively"
    echo "  ./run_red_teamer.sh info                - Show framework information"
    $PYTHON -m redteamer.cli info
else
    $PYTHON -m redteamer.cli "$@"
fi 