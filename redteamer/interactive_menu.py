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
    console.print("[bold]Interactive Red Team Evaluation[/bold]")
    console.print("Run a comprehensive red team evaluation\n")
    
    # Simplified version - in a real implementation, you'd have more options
    console.print("[yellow]Running interactive red team evaluation...[/yellow]")
    
    try:
        subprocess.run(["python", "-m", "redteamer.cli", "run"])
    except Exception as e:
        console.print(f"[bold red]Error running command:[/bold red] {str(e)}")
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
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