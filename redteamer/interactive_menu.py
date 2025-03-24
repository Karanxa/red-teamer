"""
Interactive menu for the Red Teaming Framework.
"""

import os
import sys
import inquirer
import subprocess
from typing import List, Dict, Optional, Any

from rich import print
from rich.console import Console
from rich.panel import Panel
from redteamer.models import get_all_available_models

# Initialize console for rich output
console = Console()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the Red Teaming Framework header."""
    clear_screen()
    console.print(Panel.fit(
        "[bold red]Red Teaming Framework[/bold red]\n"
        "[italic]Interactive Menu[/italic]",
        border_style="red"
    ))

def get_available_datasets() -> List[str]:
    """Get a list of available datasets."""
    datasets = []
    
    # Check examples directory
    if os.path.exists("examples"):
        for file in os.listdir("examples"):
            if file.endswith(".json"):
                datasets.append(f"examples/{file}")
    
    # Check datasets directory
    if os.path.exists("datasets"):
        for file in os.listdir("datasets"):
            if file.endswith(".json"):
                datasets.append(f"datasets/{file}")
    
    return datasets

def get_api_keys() -> Dict[str, bool]:
    """Check which API keys are set."""
    return {
        "openai": os.environ.get("OPENAI_API_KEY") is not None,
        "anthropic": os.environ.get("ANTHROPIC_API_KEY") is not None,
        "gemini": os.environ.get("GOOGLE_API_KEY") is not None
    }

def get_available_models() -> Dict[str, List[str]]:
    """Get a list of available models by provider."""
    # This will get models from the framework's configured providers
    available_models = get_all_available_models()
    
    # Fetch running Ollama models
    if "ollama" in available_models:
        # We'll use the actual models from the framework
        return available_models
    else:
        # Fallback to default models if Ollama is not available
        return {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "gemini": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "ollama": []  # Empty if no Ollama models are detected
        }

def main_menu():
    """Show the main menu and handle user selection."""
    print_header()
    
    # Check API keys
    api_keys = get_api_keys()
    api_status = []
    for provider, status in api_keys.items():
        icon = "✓" if status else "✗"
        color = "green" if status else "red"
        api_status.append(f"[{color}]{icon}[/{color}] {provider.capitalize()}")
    
    console.print("[bold]API Keys:[/bold]", ", ".join(api_status))
    console.print()
    
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Raw Scan (Static Scan)",
                "Conversation Scan (Chatbot Scan)",
                "View Scan Results",
                "Exit"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Raw Scan (Static Scan)":
        static_scan_menu()
    elif answers["action"] == "Conversation Scan (Chatbot Scan)":
        interactive_redteam_menu()
    elif answers["action"] == "View Scan Results":
        report_menu()
    elif answers["action"] == "Exit":
        console.print("[bold green]Thank you for using the Red Teaming Framework![/bold green]")
        sys.exit(0)

def static_scan_menu():
    """Menu for running a static scan."""
    print_header()
    console.print("[bold]Static Scan[/bold]")
    console.print("Run a quick scan with minimal interaction\n")
    
    # Get available models by provider
    available_models = get_all_available_models()
    api_keys = get_api_keys()
    
    # Filter models based on available API keys
    usable_providers = []
    for provider, has_key in api_keys.items():
        if has_key:
            usable_providers.append(provider)
    
    # Add Ollama if available
    if "ollama" in available_models and available_models["ollama"]:
        usable_providers.append("ollama")
    
    # Prepare model choices
    model_choices = []
    for provider in usable_providers:
        for model in available_models.get(provider, {}):
            model_choices.append(f"{provider}:{model}")
    
    # Add custom model option
    model_choices.append("Use custom model")
    
    if not model_choices or len(model_choices) == 1:  # Only custom model option
        console.print("[yellow]No models available. You can use a custom model or set up API keys.[/yellow]")
        
        # Ask if user wants to use a custom model
        use_custom = inquirer.confirm("Would you like to use a custom model?", default=True)
        if use_custom:
            run_custom_model_scan()
            return
        else:
            console.print("\n[yellow]Please set up API keys or ensure Ollama is running with models available.[/yellow]")
            console.print("Returning to main menu...")
            input("\nPress Enter to continue...")
            main_menu()
            return
    
    # Initialize prompt engine
    from redteamer.prompt_engine import PromptEngine
    prompt_engine = PromptEngine()
    intensity_levels = prompt_engine.get_intensity_levels()
    
    # Create intensity level choices
    intensity_choices = []
    for level, config in intensity_levels.items():
        intensity_choices.append(f"Level {level} - {config['prompts']} prompts using {len(config['techniques'])} techniques")
    
    questions = [
        inquirer.List(
            "model",
            message="Select a model",
            choices=model_choices
        ),
        inquirer.List(
            "intensity",
            message="Select scan intensity level",
            choices=intensity_choices
        ),
        inquirer.Confirm(
            "verbose",
            message="Enable verbose output?",
            default=False
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    # Handle custom model if selected
    if answers["model"] == "Use custom model":
        run_custom_model_scan()
        return
    
    # Parse provider and model with error handling
    try:
        model_parts = answers["model"].split(":")
        if len(model_parts) != 2:
            console.print("[red]Invalid model format. Expected 'provider:model'[/red]")
            input("\nPress Enter to continue...")
            main_menu()
            return
        provider, model = model_parts
    except Exception as e:
        console.print(f"[red]Error parsing model: {str(e)}[/red]")
        input("\nPress Enter to continue...")
        main_menu()
        return
    
    # Parse intensity level
    intensity_level = int(answers["intensity"].split()[1])
    
    # Generate prompts based on intensity level
    console.print(f"\n[bold]Generating {intensity_levels[intensity_level]['prompts']} prompts...[/bold]")
    prompts = prompt_engine.generate_prompts(intensity_level)
    
    # Save prompts to a temporary file
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"prompts": prompts}, f)
        temp_prompt_file = f.name
    
    # Build the command
    cmd = ["python", "-m", "redteamer.cli", "static_scan",
           "--provider", provider,
           "--model", model,
           "--prompts-file", temp_prompt_file]

    # Add verbose flag if selected
    if answers["verbose"]:
        cmd.append("--verbose")
    
    # Run the command
    console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
    
    # Confirm before running
    confirm = inquirer.confirm("Ready to run the scan?", default=True)
    if confirm:
        try:
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_prompt_file)
            except:
                pass
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def run_custom_model_scan(datasets, dataset_path=None, sample_size="10", verbose=False):
    """Run a scan with a custom model."""
    # If no dataset was selected previously, select one now
    if dataset_path is None:
        dataset_question = [
            inquirer.List(
                "dataset",
                message="Select a dataset",
                choices=datasets
            )
        ]
        dataset_answer = inquirer.prompt(dataset_question)
        dataset_path = dataset_answer["dataset"]
        
        # Handle dataset creation
        if dataset_path == "Create new dataset":
            console.print("[yellow]Dataset creation not implemented in this demo[/yellow]")
            dataset_path = "examples/sample_attack_vectors.json"
    
    # Get custom model details
    custom_questions = [
        inquirer.Text(
            "custom_model_name",
            message="Name for your custom model",
            default="custom-model"
        ),
        inquirer.Text(
            "custom_model",
            message="Enter curl command with {prompt} placeholder",
            default="echo 'This is a response from {prompt}'"
        )
    ]
    custom_answers = inquirer.prompt(custom_questions)
    
    # Build the command
    cmd = ["python", "-m", "redteamer.cli", "static_scan",
           "--custom-model", custom_answers["custom_model"],
           "--custom-model-name", custom_answers["custom_model_name"],
           "--dataset", dataset_path,
           "--sample-size", sample_size]
    
    # Add verbose flag if selected
    if verbose:
        cmd.append("--verbose")
    
    # Run the command
    console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
    
    # Confirm before running
    confirm = inquirer.confirm("Ready to run the scan?", default=True)
    if confirm:
        try:
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def interactive_redteam_menu():
    """Menu for running an interactive red team evaluation."""
    print_header()
    console.print("[bold]Conversation Scan[/bold]")
    console.print("Run an interactive evaluation with a model\n")
    
    questions = [
        inquirer.List(
            "action",
            message="What type of conversation scan would you like to run?",
            choices=[
                "Manual Conversation Test",
                "Conversational Red-Teaming Scanner",
                "Return to Main Menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Manual Conversation Test":
        test_model_menu()
    elif answers["action"] == "Conversational Red-Teaming Scanner":
        conversational_redteam_menu()
    elif answers["action"] == "Return to Main Menu":
        main_menu()

def conversational_redteam_menu():
    """Menu for running the conversational red-teaming scanner."""
    print_header()
    console.print("[bold]Conversational Red-Teaming Scanner[/bold]")
    console.print("Automatically test a model with adversarial dialogue\n")
    
    # Get available models by provider
    available_models = get_all_available_models()
    api_keys = get_api_keys()
    
    # Filter models based on available API keys
    usable_providers = []
    for provider, has_key in api_keys.items():
        if has_key:
            usable_providers.append(provider)
    
    # Add Ollama if available
    if "ollama" in available_models and available_models["ollama"]:
        usable_providers.append("ollama")
    
    # Add curl as an option (always available)
    usable_providers.append("curl")
    
    # Prepare model choices for each provider
    provider_choices = []
    for provider in usable_providers:
        provider_choices.append({
            "name": f"{provider.capitalize()} Model",
            "value": provider
        })
    
    # Ask for target model type
    provider_question = [
        inquirer.List(
            "provider",
            message="Select target model type",
            choices=provider_choices
        )
    ]
    
    provider_answer = inquirer.prompt(provider_question)
    provider = provider_answer["provider"]
    
    # Extract the provider value if it's a dictionary
    provider_value = provider["value"] if isinstance(provider, dict) else provider
    
    # Set up model configuration based on provider
    model_config = {}
    
    if provider_value == "curl":
        # Ask for curl command template
        curl_questions = [
            inquirer.Text(
                "curl_command",
                message="Enter curl command (use {prompt} and optional {system_prompt} placeholders)"
            ),
            inquirer.Text(
                "system_prompt",
                message="Enter optional system prompt (leave empty if not needed)"
            )
        ]
        
        curl_answers = inquirer.prompt(curl_questions)
        
        model_config = {
            "curl_command": curl_answers["curl_command"],
            "system_prompt": curl_answers["system_prompt"] if curl_answers["system_prompt"] else None
        }
        
        model_name = "Custom curl"
    elif provider_value == "ollama":
        # Ask for Ollama model
        ollama_models = available_models.get("ollama", [])
        
        if not ollama_models:
            console.print("[yellow]No Ollama models detected. Make sure Ollama is running.[/yellow]")
            console.print("Returning to main menu...")
            input("\nPress Enter to continue...")
            main_menu()
            return
        
        ollama_questions = [
            inquirer.List(
                "model",
                message="Select Ollama model",
                choices=ollama_models
            ),
            inquirer.Text(
                "system_prompt",
                message="Enter optional system prompt (leave empty if not needed)"
            )
        ]
        
        ollama_answers = inquirer.prompt(ollama_questions)
        
        model_config = {
            "model": ollama_answers["model"],
            "system_prompt": ollama_answers["system_prompt"] if ollama_answers["system_prompt"] else None,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        model_name = ollama_answers["model"]
    else:
        # Handle standard API-based models
        models = available_models.get(provider_value, [])
        
        if not models:
            console.print(f"[yellow]No models available for {provider_value}.[/yellow]")
            console.print("Returning to main menu...")
            input("\nPress Enter to continue...")
            main_menu()
            return
        
        model_questions = [
            inquirer.List(
                "model",
                message=f"Select {provider_value} model",
                choices=models
            ),
            inquirer.Text(
                "system_prompt",
                message="Enter optional system prompt (leave empty if not needed)"
            )
        ]
        
        model_answers = inquirer.prompt(model_questions)
        
        model_config = {
            "provider": provider_value,
            "model": model_answers["model"],
            "system_prompt": model_answers["system_prompt"] if model_answers["system_prompt"] else None,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        model_name = model_answers["model"]
    
    # Ask for chatbot context description
    context_question = [
        inquirer.Text(
            "chatbot_context",
            message="Enter a brief description of the chatbot/model (purpose, usage, development reasons)",
            default="An AI assistant designed to be helpful, harmless, and honest."
        )
    ]
    
    context_answer = inquirer.prompt(context_question)
    chatbot_context = context_answer["chatbot_context"]
    
    # Ask for HuggingFace model and API key
    hf_questions = [
        inquirer.Text(
            "model_id",
            message="Enter HuggingFace model ID for red-teaming (e.g., meta-llama/Llama-2-7b-chat-hf)",
            default="meta-llama/Llama-2-7b-chat-hf"
        ),
        inquirer.Text(
            "api_key",
            message="Enter HuggingFace API key (leave empty if not needed)"
        ),
        inquirer.List(
            "quant_mode",
            message="Select quantization mode for the model",
            choices=[
                "Auto (best for your hardware)",
                "8-bit (faster, more memory)",
                "4-bit (slower, less memory)",
                "CPU only (no GPU)"
            ]
        ),
        inquirer.List(
            "iterations",
            message="Select number of conversation iterations",
            choices=["5", "10", "20", "30"]
        ),
        inquirer.Confirm(
            "verbose",
            message="Enable verbose output?",
            default=False
        )
    ]
    
    hf_answers = inquirer.prompt(hf_questions)
    
    redteam_model_id = hf_answers["model_id"]
    hf_api_key = hf_answers["api_key"] if hf_answers["api_key"] else None
    quant_mode = hf_answers["quant_mode"].split()[0].lower()  # Extract first word and lowercase
    max_iterations = int(hf_answers["iterations"])
    verbose = hf_answers["verbose"]
    
    # Set up the output directory
    output_dir = "results/conversational"
    os.makedirs(output_dir, exist_ok=True)
    
    # Confirm before running
    console.print("\n[bold]Conversational Red-Teaming Configuration:[/bold]")
    console.print(f"Target: {provider_value.capitalize()} - {model_name}")
    console.print(f"Red-Teaming Model: {redteam_model_id}")
    console.print(f"Quantization Mode: {hf_answers['quant_mode']}")
    console.print(f"Iterations: {max_iterations}")
    console.print(f"Output Directory: {output_dir}")
    console.print()
    
    confirm = inquirer.confirm("Ready to run the conversational red-teaming scan?", default=True)
    
    if confirm:
        try:
            console.print("\n[bold]Launching Conversational Red-Teaming Scanner...[/bold]")
            
            # Import the conversational red-teaming module
            from redteamer.red_team.conversational_redteam import run_conversational_redteam
            
            # Run the red-teaming process
            run_conversational_redteam(
                target_model_type=provider_value,
                chatbot_context=chatbot_context,
                redteam_model_id=redteam_model_id,
                model_config=model_config,
                hf_api_key=hf_api_key,
                max_iterations=max_iterations,
                output_dir=output_dir,
                verbose=verbose,
                quant_mode=quant_mode
            )
            
        except Exception as e:
            console.print(f"\n[bold red]Error running conversational red-teaming:[/bold red] {str(e)}")
        
        input("\nPress Enter to continue...")
    
    # Return to main menu
    main_menu()

def test_model_menu():
    """Menu for testing a model with a single prompt."""
    print_header()
    console.print("[bold]Test Model[/bold]")
    console.print("Send a single prompt to a model\n")
    
    # Get available models by provider
    available_models = get_available_models()
    api_keys = get_api_keys()
    
    # Filter models based on available API keys
    usable_providers = []
    for provider, has_key in api_keys.items():
        if has_key:
            usable_providers.append(provider)
    
    if "ollama" in available_models:
        usable_providers.append("ollama")
    
    # Prepare model choices
    model_choices = []
    for provider in usable_providers:
        for model in available_models.get(provider, []):
            model_choices.append(f"{provider}:{model}")
    
    model_choices.append("Use custom model")
    
    questions = [
        inquirer.List(
            "model",
            message="Select a model",
            choices=model_choices
        ),
        inquirer.Text(
            "prompt",
            message="Enter your prompt"
        ),
        inquirer.Text(
            "system_prompt",
            message="Enter system prompt (optional)"
        ),
        inquirer.Confirm(
            "evaluate",
            message="Evaluate response for security?",
            default=True
        ),
        inquirer.Confirm(
            "verbose",
            message="Enable verbose output?",
            default=False
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    # Handle custom model if selected
    if answers["model"] == "Use custom model":
        custom_questions = [
            inquirer.Text(
                "custom_model",
                message="Enter curl command with {prompt} placeholder",
                default="echo 'This is a response from {prompt}'"
            )
        ]
        custom_answers = inquirer.prompt(custom_questions)
        
        # Not implementing custom model test in this demo
        console.print("[yellow]Custom model testing not implemented in this demo[/yellow]")
        input("\nPress Enter to return to the main menu...")
        main_menu()
        return
    
    # Parse provider and model
    provider, model = answers["model"].split(":")
    
    # Build the command
    cmd = ["python", "-m", "redteamer.cli", "test",
           "--provider", provider,
           "--model", model,
           "--prompt", answers["prompt"]]
    
    # Add system prompt if provided
    if answers["system_prompt"]:
        cmd.extend(["--system", answers["system_prompt"]])
    
    # Add evaluator if selected
    if answers["evaluate"]:
        cmd.extend(["--evaluator", "rule-based"])
    
    # Add verbose flag if selected
    if answers["verbose"]:
        cmd.append("--verbose")
    
    # Run the command
    console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
    
    # Confirm before running
    confirm = inquirer.confirm("Ready to test the model?", default=True)
    if confirm:
        try:
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def dataset_menu():
    """Menu for managing datasets."""
    print_header()
    console.print("[bold]Manage Datasets[/bold]")
    
    # Get available datasets
    datasets = get_available_datasets()
    
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Create a new dataset",
                "Add vector to existing dataset",
                "View dataset statistics",
                "Return to main menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Create a new dataset":
        # Run dataset create command
        try:
            subprocess.run(["python", "-m", "redteamer.cli", "dataset", "create"])
        except Exception as e:
            console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    elif answers["action"] == "Add vector to existing dataset":
        if not datasets:
            console.print("[yellow]No datasets found. Please create a dataset first.[/yellow]")
        else:
            dataset_question = [
                inquirer.List(
                    "dataset",
                    message="Select a dataset",
                    choices=datasets
                )
            ]
            dataset_answer = inquirer.prompt(dataset_question)
            
            # Run add-vector command
            try:
                subprocess.run(["python", "-m", "redteamer.cli", "dataset", "add-vector", dataset_answer["dataset"]])
            except Exception as e:
                console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    elif answers["action"] == "View dataset statistics":
        if not datasets:
            console.print("[yellow]No datasets found.[/yellow]")
        else:
            dataset_question = [
                inquirer.List(
                    "dataset",
                    message="Select a dataset",
                    choices=datasets
                )
            ]
            dataset_answer = inquirer.prompt(dataset_question)
            
            # Run stats command
            try:
                subprocess.run(["python", "-m", "redteamer.cli", "dataset", "stats", dataset_answer["dataset"]])
            except Exception as e:
                console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    elif answers["action"] == "Return to main menu":
        main_menu()
        return
    
    # Wait for user to press enter before returning to dataset menu
    input("\nPress Enter to continue...")
    dataset_menu()

def report_menu():
    """Menu for generating reports."""
    print_header()
    console.print("[bold]Generate Reports[/bold]")
    
    # Get available results
    results = []
    if os.path.exists("results"):
        for file in os.listdir("results"):
            if file.endswith(".json"):
                results.append(f"results/{file}")
    
    if not results:
        console.print("[yellow]No results found. Run a scan or evaluation first.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        main_menu()
        return
    
    questions = [
        inquirer.List(
            "results_file",
            message="Select results file",
            choices=results
        ),
        inquirer.List(
            "format",
            message="Select report format",
            choices=["markdown", "json", "csv", "pdf"]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    # Run report generate command
    cmd = ["python", "-m", "redteamer.cli", "report", "generate",
           answers["results_file"],
           "--format", answers["format"]]
    
    console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def view_documentation():
    """View framework documentation."""
    print_header()
    console.print("[bold]Documentation[/bold]")
    console.print(Panel.fit(
        "[bold]Commands Overview[/bold]\n\n"
        "[cyan]static_scan[/cyan] - Run a quick scan with minimal interaction\n"
        "[cyan]run[/cyan] - Run a comprehensive red team evaluation\n"
        "[cyan]test[/cyan] - Test a model with a single prompt\n"
        "[cyan]dataset create[/cyan] - Create a new dataset\n"
        "[cyan]dataset add-vector[/cyan] - Add a vector to a dataset\n"
        "[cyan]dataset stats[/cyan] - View dataset statistics\n"
        "[cyan]report generate[/cyan] - Generate a report from results\n",
        title="Help",
        border_style="blue"
    ))
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def run():
    """Run the interactive menu."""
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Program interrupted by user. Exiting...[/bold yellow]")
        sys.exit(0)

if __name__ == "__main__":
    run() 