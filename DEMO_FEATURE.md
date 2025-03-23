# Contextual Chatbot Red Teaming Demo Feature

## Overview

We've implemented a new interactive demo feature in the Red Teaming Framework that allows users to experience the contextual chatbot red teaming process with a real-time visualization dashboard. This feature is designed to make it easy for users to understand how the red teaming system works, see the types of adversarial prompts generated, and analyze the results of the evaluation.

## Components Implemented

1. **Streamlit Dashboard (`redteamer/demo/streamlit_app.py`)**:
   - Real-time visualization of the red teaming process
   - Progress tracking and statistics
   - Detailed view of prompts, responses, and evaluations
   - Interactive elements for exploring results

2. **Demo Engine (`redteamer/demo/contextual_demo.py`)**:
   - Integration with Gemini API for model testing
   - Automatic Streamlit server management
   - Background processing with real-time dashboard updates
   - Interactive command line interface with parameter handling

3. **CLI Integration**:
   - Added demo sub-command to the main CLI app
   - Command line arguments for API key, context file, and test parameters
   - Seamless connection to the existing contextual red teaming functionality

4. **Documentation**:
   - Added demo feature documentation to README
   - Created specific README for the demo module
   - Updated project requirements to include Streamlit

## Key Features

- **Real-time visualization**: Users can see the red teaming process as it happens
- **Gemini model integration**: Quick setup with Google's Gemini model for demonstration
- **Contextual prompt generation**: Uses the karanxa/dravik model to generate context-aware adversarial prompts
- **Interactive dashboard**: Filter, sort, and explore results through the Streamlit interface
- **Statistical summary**: Quick overview of attack success rate and categories

## How It Works

1. User runs the demo command and provides a Gemini API key
2. The system starts a local Streamlit server and provides a URL to the user
3. The dashboard initializes and waits for results
4. In the background, the system:
   - Generates contextual adversarial prompts tailored to the provided context
   - Sends these prompts to the Gemini API
   - Evaluates the responses for security vulnerabilities
   - Sends results to the dashboard in real-time
5. The user can watch the process unfold and explore results as they come in

## Usage

```bash
# Basic usage
python -m redteamer.cli demo run

# With API key and custom settings
python -m redteamer.cli demo run --api-key YOUR_API_KEY --num-prompts 20 --category jailbreak
```

## Requirements

- Streamlit (for visualization dashboard)
- Google Generative AI Python library (for Gemini model access)
- Valid Gemini API key
- Internet connection for API calls

## Future Enhancements

Potential future improvements to the demo feature:
- Support for multiple model types beyond Gemini
- Customizable evaluation criteria
- Export of results from the dashboard
- Comparison of different models side-by-side
- More detailed analysis visualizations

## Conclusion

This demo feature significantly enhances the usability of the Red Teaming Framework by providing a visual, interactive way to understand the contextual red teaming process. It serves educational purposes, allows for quick demonstrations, and helps users grasp the concepts of LLM security testing in an intuitive way. 