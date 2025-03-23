# Contextual Chatbot Red Teaming Demo Implementation

## Overview

We've successfully implemented a comprehensive demo feature for contextual chatbot red teaming with real-time visualization capabilities. This feature allows users to experience the red teaming process through an interactive Streamlit dashboard, making it easier to understand how the system works and to demonstrate its capabilities.

## Key Components Implemented

1. **Streamlit Dashboard (`redteamer/demo/streamlit_app.py`)**
   - Real-time visualization of red teaming progress
   - Interactive results table and detailed view
   - Progress tracking and statistics
   - Event-based communication with the main process

2. **Demo Engine (`redteamer/demo/contextual_demo.py`)**
   - Integration with Gemini API for live model testing
   - Sample data generator for API-free demonstrations
   - Subprocess management for the Streamlit server
   - Thread management for background processing

3. **CLI Integration**
   - Added demo subcommand to the main CLI
   - Command-line options for customization
   - Support for both API key and sample modes

4. **Convenience Scripts**
   - `run_demo_sample.py` - Quick launcher for sample mode
   - `test_demo.py` - Direct function call for testing

## Features

- **Live Model Testing**: Test Gemini models directly in the UI
- **Sample Mode**: Demonstrate functionality without requiring API keys
- **Real-time Updates**: Watch the red teaming process as it happens
- **Visual Statistics**: See attack success rates and categories
- **Detailed Analysis**: Explore individual prompts, responses, and evaluations

## Sample Mode

The sample mode is a key innovation that allows users to experience the red teaming process without requiring any API keys. It includes:

- Realistic examples of various attack vectors
- Mix of successful and unsuccessful attacks
- Representation of different attack categories
- Simulated response delays for realism

## User Experience

We've designed the user experience to be as smooth as possible:
1. User runs the demo command (with or without an API key)
2. A local Streamlit server starts automatically
3. User opens the provided URL in a browser
4. The dashboard shows real-time updates as the red teaming progresses
5. User can explore results and statistics interactively

## Documentation

- Updated main README with demo information
- Created dedicated README for the demo module
- Added comprehensive documentation in the code
- Created simple wrapper scripts with clear documentation

## Future Enhancements

Potential future improvements:
1. Support for additional model types beyond Gemini
2. Filter and search capabilities in the dashboard
3. Export and save functionality for results
4. Comparison view for different models or configurations
5. More detailed visualization of attack patterns

## Usage Examples

```bash
# With API key
python -m redteamer.cli demo run --api-key YOUR_API_KEY

# With sample data (no API key needed)
python -m redteamer.cli demo run --sample

# Using the wrapper script
python run_demo_sample.py

# With custom options
python -m redteamer.cli demo run --api-key YOUR_API_KEY --num-prompts 20 --category jailbreak
```

## Conclusion

The implemented demo feature significantly enhances the usability and demonstration capabilities of the red teaming framework. It provides an intuitive, visual way to understand the red teaming process and helps users grasp the concepts of LLM security testing. 