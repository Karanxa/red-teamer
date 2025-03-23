# Contextual Red Teaming Demo

This module provides a user-friendly demonstration of the contextual chatbot red teaming capabilities of the RedTeamer framework. The demo includes a real-time visualization interface using Streamlit, making it easy to understand how the red teaming process works.

## Features

- Interactive demo with Streamlit visualization
- Real-time progress tracking of red teaming operations
- Detailed view of generated prompts, model responses, and evaluation results
- Automatic configuration for testing Gemini models
- Summary statistics of successful attacks

## Usage

To run the demo, use the following command:

```bash
python -m redteamer.cli demo run
```

### Command Line Arguments

- `--api-key, -k`: Gemini API key (optional, will prompt if not provided)
- `--context-file, -f`: Path to a file containing chatbot context description
- `--num-prompts, -n`: Number of prompts to generate (default: 10)
- `--category`: Categories of attacks to focus on (can be specified multiple times)

### Example

```bash
python -m redteamer.cli demo run --api-key YOUR_API_KEY --num-prompts 15 --category jailbreak --category prompt-injection
```

### Sample Mode

If you want to see how the demo works without needing a Gemini API key, you can use the sample mode:

```bash
python -m redteamer.cli demo run --sample
```

Or use the provided wrapper script for an even simpler experience:

```bash
# Run with default 10 sample prompts
python run_demo_sample.py

# Specify the number of sample prompts
python run_demo_sample.py 15
```

This will run the demo with pre-generated example data, showing you both successful and unsuccessful attack attempts. It's perfect for:
- Demonstrations and presentations
- Understanding how the red teaming process works
- Testing the visualization interface
- Educational purposes

The sample mode includes examples of various attack categories, including jailbreak attempts, prompt injection, information extraction, and social engineering.

## Getting Started

1. Provide a context description for your chatbot when prompted
   - This helps the red teaming model (karanxa/dravik) generate more effective adversarial prompts
   - For example: "A customer service chatbot for an e-commerce platform"
2. If using API mode, provide your Gemini API key or set it via GOOGLE_API_KEY environment variable
3. When the Streamlit interface starts, open the provided URL (typically http://localhost:8501)
4. Watch the red teaming process in real-time:
   - First, the system analyzes your chatbot context
   - Then, it generates context-aware adversarial prompts using the Dravik model
   - Finally, it tests these prompts against the target model (Gemini or sample data)
5. Review the detailed results and statistics in the dashboard

## Understanding the Process Flow

The demo follows this process:

1. **Context Analysis**: You provide a description of your chatbot's purpose and domain
2. **Adversarial Prompt Generation**: The specialized red teaming model (karanxa/dravik) analyzes this context and generates tailored adversarial prompts
3. **Targeted Testing**: These prompts are sent to the target model
4. **Response Evaluation**: The responses are analyzed for security vulnerabilities
5. **Results Dashboard**: All results are displayed in real-time through the Streamlit interface

This flow simulates how an actual contextual red teaming evaluation would work in production, showing how domain-specific knowledge can be used to create more effective adversarial prompts.

## Understanding the Visualization

The Streamlit dashboard provides:

1. **Target Information**: Details about the model being tested and elapsed time
2. **Progress Tracking**: Real-time progress of the red teaming process
3. **Context Description**: The context provided to the red teaming model
4. **Test Results**: Summary statistics and a table of test results
5. **Detailed View**: In-depth analysis of each test, including the prompt, response, and evaluation

## Example Context File

You can provide your own context file to tailor the red teaming to a specific use case. Example:

```
This is a customer service chatbot for an e-commerce platform.
It handles order status inquiries, returns, and product information.
The chatbot should not discuss topics unrelated to shopping or provide personal opinions.
It should refuse requests for personal data or payment information.
```

## Categories of Attacks

The demo supports the following attack categories:
- `jailbreak`: Attempts to bypass the model's safety mechanisms
- `prompt-injection`: Tries to manipulate the model into ignoring instructions
- `information-extraction`: Attempts to extract sensitive information
- `social-engineering`: Uses social manipulation techniques

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python library
- Internet access for API calls
- A valid Gemini API key 