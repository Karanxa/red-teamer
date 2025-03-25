"""
Interactive menu for the Red Teaming Framework.
"""

import os
import sys
import inquirer
import subprocess
from typing import List, Dict, Optional, Any
from pathlib import Path

from rich import print
from rich.console import Console
from rich.panel import Panel
import questionary
from rich.table import Table
import platform
import logging
import tempfile
from datetime import datetime
from rich.prompt import Prompt, IntPrompt, Confirm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redteamer")

# Initialize console for rich output
console = Console()

def get_valid_index(prompt: str, max_idx: int) -> int:
    """Get a valid index from user input.
    
    Args:
        prompt: The prompt to display to the user
        max_idx: The maximum valid index (exclusive)
        
    Returns:
        A valid index between 0 and max_idx-1
    """
    while True:
        try:
            # Get user input with 1-based indexing
            idx = int(input(prompt)) - 1
            
            # Check if the index is valid
            if 0 <= idx < max_idx:
                return idx
            else:
                console.print(f"[red]Please enter a number between 1 and {max_idx}.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

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

def get_api_keys(check_keys=False) -> Dict[str, bool]:
    """Check which API keys are set using the API key manager.
    
    Args:
        check_keys: If True, actually check API keys. If False, return dummy values.
    """
    # If check_keys is False, return dummy values without actually checking
    if not check_keys:
        return {
            "openai": False,
            "anthropic": False,
            "gemini": False,
            "huggingface": False
        }
    
    # Import on demand to avoid API key checks at startup
    from redteamer.utils.api_key_manager import get_api_key_manager
    
    # Get the API key manager
    api_key_manager = get_api_key_manager()
    
    # Get key status for each provider
    return {
        "openai": api_key_manager.get_key("openai") is not None,
        "anthropic": api_key_manager.get_key("anthropic") is not None,
        "gemini": api_key_manager.get_key("gemini") is not None,
        "huggingface": api_key_manager.get_key("huggingface") is not None
    }

def get_ollama_models() -> List[str]:
    """Get a list of available ollama models.
    
    Returns:
        A list of available ollama model names or an empty list if ollama is not available
    """
    try:
        # Run ollama list and capture the output
        result = subprocess.run(
            ["ollama", "list"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Parse the model names from the output
        models = []
        lines = result.stdout.strip().split('\n')
        
        # Skip the header line if it exists
        if lines and not lines[0].startswith('NAME'):
            start_idx = 0
        else:
            start_idx = 1
            
        # Extract model names from each line
        for line in lines[start_idx:]:
            if line.strip():
                # Model name is the first column
                model_name = line.strip().split()[0]
                if model_name:
                    models.append(model_name)
        
        return models
    except Exception as e:
        logger.debug(f"Error fetching ollama models: {str(e)}")
        return []

def get_available_models(provider=None) -> Dict[str, List[str]]:
    """Get a list of available models by provider.
    
    Args:
        provider: Optional provider to get models for. If None, gets models from all providers.
    """
    # Import here to avoid checking API keys at startup
    from redteamer.models import get_all_available_models
    
    # Get models from the specified provider or all providers
    available_models = get_all_available_models(provider=provider)
    
    # If provider is ollama, directly fetch running ollama models
    if provider == "ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            # Add the ollama models to the available_models dictionary
            available_models = {"ollama": {model: {} for model in ollama_models}}
    
    # If we have available models, return them
    if available_models:
        return available_models
    else:
        # Fallback to default models if no models are available
        fallback = {}
        if not provider or provider == "openai":
            fallback["openai"] = ["gpt-4", "gpt-3.5-turbo"]
        if not provider or provider == "anthropic":
            fallback["anthropic"] = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        if not provider or provider == "gemini":
            fallback["gemini"] = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
        if not provider or provider == "ollama":
            # Try to get ollama models directly if requested
            ollama_models = get_ollama_models()
            if ollama_models:
                fallback["ollama"] = ollama_models
            else:
                fallback["ollama"] = []  # Empty if no Ollama models are detected
        return fallback

def manage_api_keys_menu():
    """Display menu for managing API keys."""
    console.clear()
    console.print("[bold blue]API Key Management Menu[/bold blue]")
    console.print("Manage your API keys for different providers.")
    console.print()
    
    # Import API key manager only when needed
    from redteamer.utils.api_key_manager import get_api_key_manager
    
    # Get the API key manager
    api_key_manager = get_api_key_manager()
    
    # Get all providers with their status
    providers = api_key_manager.list_providers()
    
    # Display current API keys status
    table = Table(title="Current API Keys")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key Preview", style="dim")
    
    for provider in providers:
        status = "[green]✓ Available[/green]" if provider["has_key"] else "[red]✗ Not Set[/red]"
        key_preview = provider["key_preview"] if provider["has_key"] else ""
        table.add_row(
            provider["display_name"],
            status,
            key_preview
        )
    
    console.print(table)
    console.print()
    
    # List options for key management
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Set a new API key",
                "Delete an API key",
                "Return to main menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Set a new API key":
        # Get provider choices (excluding aliases)
        provider_choices = [p["display_name"] for p in providers]
        
        # Ask which provider
        provider_question = [
            inquirer.List(
                "provider",
                message="Select a provider:",
                choices=provider_choices
            )
        ]
        
        provider_answer = inquirer.prompt(provider_question)
        selected_display_name = provider_answer["provider"]
        
        # Find the provider id from the display name
        selected_provider = None
        for p in providers:
            if p["display_name"] == selected_display_name:
                selected_provider = p["provider"]
                break
        
        if not selected_provider:
            console.print("[red]Error: Provider not found.[/red]")
            return
        
        # Ask for API key
        key = questionary.password(f"Enter your {selected_display_name} API key:").unsafe_ask()
        
        if not key:
            console.print("[yellow]No key entered. Operation cancelled.[/yellow]")
            return
        
        # Set the key
        success = api_key_manager.set_key(selected_provider, key)
        if success:
            console.print(f"[green]✓[/green] {selected_display_name} API key has been set successfully.")
        else:
            console.print(f"[red]Error: Failed to set {selected_display_name} API key.[/red]")
        
        # Return to API key management menu
        input("\nPress Enter to continue...")
        manage_api_keys_menu()
        
    elif answers["action"] == "Delete an API key":
        # Get providers with keys
        available_providers = [p for p in providers if p["has_key"]]
        
        if not available_providers:
            console.print("[yellow]No API keys are currently set.[/yellow]")
            input("\nPress Enter to continue...")
            manage_api_keys_menu()
            return
        
        # Get provider choices
        provider_choices = [p["display_name"] for p in available_providers]
        
        # Ask which provider
        provider_question = [
            inquirer.List(
                "provider",
                message="Select a provider to delete key:",
                choices=provider_choices
            )
        ]
        
        provider_answer = inquirer.prompt(provider_question)
        selected_display_name = provider_answer["provider"]
        
        # Find the provider id from the display name
        selected_provider = None
        for p in available_providers:
            if p["display_name"] == selected_display_name:
                selected_provider = p["provider"]
                break
        
        if not selected_provider:
            console.print("[red]Error: Provider not found.[/red]")
            return
        
        # Confirm deletion
        confirm = questionary.confirm(f"Are you sure you want to delete the {selected_display_name} API key?").unsafe_ask()
        
        if not confirm:
            console.print("[yellow]Operation cancelled.[/yellow]")
            # Return to API key management menu
            input("\nPress Enter to continue...")
            manage_api_keys_menu()
            return
        
        # Delete the key
        success = api_key_manager.delete_key(selected_provider)
        if success:
            console.print(f"[green]✓[/green] {selected_display_name} API key has been deleted.")
        else:
            console.print(f"[red]Error: Failed to delete {selected_display_name} API key.[/red]")
        
        # Return to API key management menu
        input("\nPress Enter to continue...")
        manage_api_keys_menu()
    
    elif answers["action"] == "Return to main menu":
        main_menu()

def main_menu():
    """Display the main menu."""
    print_header()
    console.print("Red Teamer Framework Menu")
    console.print("Use this menu to perform various red teaming tasks on language models.")
    console.print("You can set up API keys through the 'Manage API Keys' option when needed.")
    console.print()
    
    # Get API keys status without checking them
    api_keys = get_api_keys(check_keys=False)
    
    # List menu options
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Raw Scan (Static Scan)",
                "Conversation Scan (Chatbot Scan)",
                "View Scan Results",
                "Delete Scan Results",
                "Manage API Keys",
                "Exit"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Raw Scan (Static Scan)":
        static_scan_menu()
    elif answers["action"] == "Conversation Scan (Chatbot Scan)":
        conversational_redteam_menu()
    elif answers["action"] == "View Scan Results":
        view_results_menu()
    elif answers["action"] == "Delete Scan Results":
        delete_results_menu()
    elif answers["action"] == "Manage API Keys":
        manage_api_keys_menu()
    elif answers["action"] == "Exit":
        console.print("[bold green]Thank you for using the Red Teamer Framework![/bold green]")
        sys.exit(0)

def static_scan_menu():
    """Display menu for static scan of models."""
    console.clear()
    console.print("[bold blue]Static Scan Menu[/bold blue]")
    console.print("This tool tests a model with generated adversarial prompts and evaluates responses.")
    console.print()
    
    # Define available providers
    usable_providers = ["openai", "anthropic", "gemini", "ollama"]
    custom_provider = "custom"
    
    # Check if ollama is available without loading models
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ollama_available = True
    except:
        ollama_available = False
    
    # Initialize choices
    provider_choice = None
    model_choice = None
    custom_model = None
    custom_model_name = None
    
    # Show provider selection - don't check API keys yet
    console.print("[bold]Available Model Providers:[/bold]")
    
    # Prepare provider choices for inquirer
    provider_choices = ["openai", "anthropic", "gemini"]
    if ollama_available:
        provider_choices.append("ollama")
    provider_choices.append(custom_provider)
    
    # Ask user to select a provider using inquirer
    provider_question = [
        inquirer.List(
            "provider",
            message="Select a model provider",
            choices=provider_choices
        )
    ]
    
    provider_answer = inquirer.prompt(provider_question)
    provider_choice = provider_answer["provider"]
    
    # Handle custom provider
    if provider_choice == custom_provider:
        custom_model_name = Prompt.ask("Enter a name for your custom model", default="custom-model")
        custom_model = Prompt.ask(
            "Enter your custom model curl command (use {prompt} as placeholder)",
            default="curl -X POST http://localhost:8080/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"{prompt}\"}'"
        )
        
        if "{prompt}" not in custom_model:
            console.print("[bold red]Error: Custom model command must contain {prompt} placeholder.[/bold red]")
            return
    else:
        # Check API key for the selected provider (only check the one that's needed)
        # Import API key manager only when needed
        from redteamer.utils.api_key_manager import get_api_key_manager
        api_key_manager = get_api_key_manager()
        
        # Only check API key for non-Ollama cloud providers
        if provider_choice in ["openai", "anthropic", "gemini"]:
            api_key = api_key_manager.get_key(provider_choice)
            if not api_key:
                console.print(f"[bold yellow]No API key found for {provider_choice}.[/bold yellow]")
                set_key = Confirm.ask(f"Would you like to set an API key for {provider_choice} now?")
                
                if set_key:
                    api_key = Prompt.ask(f"Enter your {provider_choice.upper()} API key")
                    api_key_manager.set_key(provider_choice, api_key)
                    console.print(f"[green]API key for {provider_choice} has been set.[/green]")
                else:
                    console.print("[yellow]Operation cancelled. API key is required.[/yellow]")
                    return
        
        # Now fetch models for the selected provider
        console.print(f"\n[bold]Fetching models for {provider_choice}...[/bold]")
        
        # Import models only when needed
        from redteamer.models import get_all_available_models
        provider_models = get_all_available_models(provider=provider_choice).get(provider_choice, {})
        
        if not provider_models:
            console.print(f"[bold red]No models available for {provider_choice}[/bold red]")
            return
        
        # Show model selection
        console.print(f"\n[bold]Available {provider_choice} models:[/bold]")
        model_names = list(provider_models.keys())
        
        # Use inquirer for model selection
        model_question = [
            inquirer.List(
                "model",
                message=f"Select a {provider_choice} model",
                choices=model_names
            )
        ]
        
        model_answer = inquirer.prompt(model_question)
        model_choice = model_answer["model"]
    
    # Ask for number of prompts
    num_prompts = IntPrompt.ask(
        "Number of adversarial prompts to generate", 
        default=10
    )
    
    # Ask for confirmation
    if provider_choice == custom_provider:
        run_message = f"Run static scan with custom model '{custom_model_name}'?"
    else:
        run_message = f"Run static scan with {provider_choice}/{model_choice}?"
    
    if Confirm.ask(run_message):
        # Launch static scan
        args = ["python", "-m", "redteamer.static_scan_launcher"]
        
        if provider_choice == custom_provider:
            args.extend(["--custom-model", custom_model])
            args.extend(["--custom-model-name", custom_model_name])
        else:
            args.extend(["--provider", provider_choice])
            args.extend(["--model", model_choice])
        
        args.extend(["--num-prompts", str(num_prompts)])
        
        # Launch the static scan
        subprocess.run(args)
    else:
        console.print("[yellow]Scan cancelled[/yellow]")

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

def view_results_menu():
    """Display menu for viewing scan results."""
    print_header()
    console.print("[bold]View Scan Results[/bold]")
    console.print("This tool helps you view and analyze the results of previous scans.\n")
    
    # Get all results
    results_dir = Path("results")
    if not results_dir.exists():
        console.print("[yellow]No results directory found.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        return main_menu()
    
    # Get all result files
    result_files = []
    
    # Get static scan results
    static_results = list(results_dir.glob("static_scan_*.json"))
    for result in static_results:
        result_files.append(("Static", result))
    
    # Get conversational scan results
    convo_results = list(results_dir.glob("conversational_*.json"))
    for result in convo_results:
        result_files.append(("Conversation", result))
    
    if not result_files:
        console.print("[yellow]No result files found.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        return main_menu()
    
    # Display results
    console.print("[bold]Available Results:[/bold]")
    
    table = Table()
    table.add_column("#", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Filename", style="green")
    table.add_column("Date", style="yellow")
    table.add_column("Size", style="blue")
    
    for i, (result_type, result_path) in enumerate(result_files):
        # Get file stats
        stats = result_path.stat()
        # Format date
        date = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
        # Format size
        if stats.st_size < 1024:
            size = f"{stats.st_size} B"
        elif stats.st_size < 1024 * 1024:
            size = f"{stats.st_size / 1024:.1f} KB"
        else:
            size = f"{stats.st_size / (1024 * 1024):.1f} MB"
        
        table.add_row(
            str(i+1),
            result_type,
            result_path.name,
            date,
            size
        )
    
    console.print(table)
    console.print()
    
    # Options
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "View a result",
                "Generate a report",
                "Return to main menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "View a result":
        # Ask which result to view
        idx = get_valid_index("Enter the number of the result to view: ", len(result_files) + 1)
        result_type, result_path = result_files[idx]
        
        # Use the system's default application to open the result file
        try:
            # Platform-specific open command
            if platform.system() == "Windows":
                os.startfile(result_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", result_path])
            else:  # Linux and other Unix-like
                subprocess.run(["xdg-open", result_path])
            
            console.print(f"[green]✓[/green] Opened {result_path.name}")
        except Exception as e:
            console.print(f"[red]Error opening file:[/red] {str(e)}")
            
            # Fallback to printing the file content
            try:
                with open(result_path, "r") as f:
                    content = f.read()
                console.print(f"[bold]Contents of {result_path.name}:[/bold]")
                console.print(content)
            except Exception as e:
                console.print(f"[red]Error reading file:[/red] {str(e)}")
        
    elif answers["action"] == "Generate a report":
        # Generate a report from a result file
        # Ask which result to use
        idx = get_valid_index("Enter the number of the result to generate a report from: ", len(result_files) + 1)
        result_type, result_path = result_files[idx]
        
        # Ask for report format
        format_question = [
            inquirer.List(
                "format",
                message="Select report format",
                choices=["markdown", "json", "csv", "pdf"]
            )
        ]
        
        format_answer = inquirer.prompt(format_question)
        report_format = format_answer["format"]
        
        # Generate the report using the CLI
        try:
            output_dir = "reports"
            os.makedirs(output_dir, exist_ok=True)
            
            # Default output filename based on result filename
            output_filename = f"{output_dir}/{result_path.stem}_report.{report_format}"
            
            # Generate the report using the CLI report command
            if report_format == "pdf":
                # PDF requires markdown as an intermediate step
                md_output = f"{output_dir}/{result_path.stem}_report.md"
                
                # Generate markdown first
                subprocess.run([
                    "python", "-m", "redteamer.cli", "report", "generate",
                    str(result_path),
                    "--output", md_output,
                    "--format", "markdown",
                    "--non-interactive"
                ])
                
                # Convert markdown to PDF
                try:
                    subprocess.run([
                        "python", "-m", "redteamer.cli", "report", "generate",
                        str(result_path),
                        "--output", output_filename,
                        "--format", "pdf",
                        "--non-interactive"
                    ])
                except Exception:
                    console.print("[yellow]PDF generation may require additional dependencies.[/yellow]")
                    console.print("[yellow]Using markdown format instead.[/yellow]")
                    output_filename = md_output
            else:
                # Generate report directly
                subprocess.run([
                    "python", "-m", "redteamer.cli", "report", "generate",
                    str(result_path),
                    "--output", output_filename,
                    "--format", report_format,
                    "--non-interactive"
                ])
            
            console.print(f"[green]✓[/green] Report generated: {output_filename}")
            
            # Ask if user wants to open the report
            open_report = Confirm.ask("Would you like to open the generated report?")
            if open_report:
                try:
                    # Platform-specific open command
                    if platform.system() == "Windows":
                        os.startfile(output_filename)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", output_filename])
                    else:  # Linux and other Unix-like
                        subprocess.run(["xdg-open", output_filename])
                except Exception as e:
                    console.print(f"[red]Error opening report:[/red] {str(e)}")
        
        except Exception as e:
            console.print(f"[red]Error generating report:[/red] {str(e)}")
    
    # Return to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def conversational_redteam_menu():
    """Menu for running conversational red teaming."""
    print_header()
    console.print("[bold]Conversational Red Teaming[/bold]")
    console.print("Test a model with an adaptive conversation\n")
    
    # Check if ollama is available
    ollama_models = get_ollama_models()
    ollama_available = len(ollama_models) > 0
    
    # Prepare provider choices
    provider_choices = ["openai", "anthropic", "gemini"]
    if ollama_available:
        provider_choices.append("ollama")
    provider_choices.append("custom")
    
    # Ask user to select a provider using inquirer
    provider_question = [
        inquirer.List(
            "provider",
            message="Select a model provider",
            choices=provider_choices
        )
    ]
    
    provider_answer = inquirer.prompt(provider_question)
    selected_provider = provider_answer["provider"]
    
    # If custom model is selected, handle it directly
    if selected_provider == "custom":
        custom_questions = [
            inquirer.Text(
                "model_name",
                message="Enter a name for your custom model",
                default="custom-model"
            ),
            inquirer.Text(
                "curl_command",
                message="Enter curl command with {prompt} placeholder",
                default="curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"llama3\",\"prompt\":\"{prompt}\"}' | jq -r '.response'"
            ),
            inquirer.Text(
                "context",
                message="Enter chatbot context (e.g., 'You are a helpful assistant')",
                default="You are a helpful AI assistant."
            ),
            inquirer.Text(
                "iterations",
                message="Enter maximum iterations",
                default="10"
            ),
            inquirer.Confirm(
                "verbose",
                message="Enable verbose output?",
                default=False
            )
        ]
        
        custom_answers = inquirer.prompt(custom_questions)
        
        # Configure the custom model
        curl_command = custom_answers["curl_command"]
        context = custom_answers["context"]
        model = custom_answers["model_name"]
        target_type = "custom"
        provider = None
        
        try:
            max_iterations = int(custom_answers["iterations"])
        except:
            max_iterations = 10
            
        verbose = custom_answers["verbose"]
        
    else:
        # For regular providers, check API key only for the selected provider
        # Import API key manager only when needed
        from redteamer.utils.api_key_manager import get_api_key_manager
        api_key_manager = get_api_key_manager()
        
        # Check if API key is available for non-ollama providers
        if selected_provider in ["openai", "anthropic", "gemini"]:
            api_key = api_key_manager.get_key(selected_provider)
            
            if not api_key:
                console.print(f"[bold yellow]No API key found for {selected_provider}.[/bold yellow]")
                set_key_now = inquirer.confirm(
                    f"Would you like to set an API key for {selected_provider} now?",
                    default=True
                )
                
                if set_key_now:
                    api_key = inquirer.text(f"Enter your {selected_provider.upper()} API key:")
                    api_key_manager.set_key(selected_provider, api_key)
                    console.print(f"[green]API key for {selected_provider} has been set.[/green]")
                else:
                    console.print("[yellow]Operation cancelled. API key is required.[/yellow]")
                    return main_menu()
        
        # Now fetch models for the selected provider
        console.print(f"\n[bold]Fetching models for {selected_provider}...[/bold]")
        
        # Get available models
        available_models = get_available_models(provider=selected_provider)
        
        # For ollama, use the models we fetched earlier
        if selected_provider == "ollama" and ollama_models:
            provider_models = ollama_models
        else:
            provider_models = list(available_models.get(selected_provider, {}).keys())
        
        if not provider_models:
            console.print(f"[bold red]No models available for {selected_provider}[/bold red]")
            return main_menu()
            
        # Use inquirer for model selection
        model_question = [
            inquirer.List(
                "model",
                message=f"Select a {selected_provider} model",
                choices=provider_models
            ),
            inquirer.Text(
                "context",
                message="Enter chatbot context (e.g., 'You are a helpful assistant')",
                default="You are a helpful AI assistant."
            ),
            inquirer.Text(
                "redteam_model",
                message="Enter red teaming model ID (optional, leave blank for default)",
                default=""
            ),
            inquirer.Text(
                "iterations",
                message="Enter maximum iterations",
                default="10"
            ),
            inquirer.Confirm(
                "verbose",
                message="Enable verbose output?",
                default=False
            )
        ]
        
        model_answers = inquirer.prompt(model_question)
        
        # Configure selected model
        provider = selected_provider
        model = model_answers["model"]
        context = model_answers["context"]
        redteam_model = model_answers["redteam_model"] if model_answers["redteam_model"] else None
        target_type = provider
        curl_command = None
        
        try:
            max_iterations = int(model_answers["iterations"])
        except:
            max_iterations = 10
            
        verbose = model_answers["verbose"]
    
    # Ask the user about interface preference
    console.print("\n[bold]Interface Preference[/bold]")
    interface_question = [
        inquirer.List(
            "interface",
            message="Select interface type",
            choices=[
                "Streamlit UI - Provides interactive visualization and real-time progress",
                "Command Line - Simpler, text-based output without browser"
            ]
        )
    ]
    
    interface_answer = inquirer.prompt(interface_question)
    # Get just the first word (Streamlit or Command)
    interface_choice = interface_answer["interface"].split()[0].lower()
    use_streamlit = interface_choice == "streamlit"
    
    # Confirm before running
    console.print("\n[bold]Conversational Red Teaming Configuration:[/bold]")
    if curl_command:
        console.print(f"Custom Model Command: {curl_command}")
    else:
        console.print(f"Provider: {provider}")
        console.print(f"Model: {model}")
    console.print(f"Chatbot Context: {context}")
    
    if redteam_model:
        console.print(f"Red Teaming Model: {redteam_model}")
    else:
        console.print("Red Teaming Model: [italic](using default)[/italic]")
    
    console.print(f"Maximum Iterations: {max_iterations}")
    console.print(f"Interface: [green]{'Streamlit UI' if use_streamlit else 'Command Line'}[/green]")
    console.print(f"Verbose: {'Yes' if verbose else 'No'}")
    console.print()
    
    if use_streamlit:
        confirm = inquirer.confirm("Ready to launch the Streamlit interface for conversational red teaming?", default=True)
    else:
        confirm = inquirer.confirm("Ready to run conversational red teaming in command line?", default=True)
    
    if confirm:
        try:
            if use_streamlit:
                console.print("\n[bold]Launching Streamlit Interface...[/bold]")
                console.print("[green]The red teaming process will run in the Streamlit interface that is being launched.[/green]")
                console.print("[green]Streamlit provides a better visual experience and real-time updates of the process.[/green]")
                console.print("[yellow]Note: If the model fails to load, the system will automatically try smaller fallback models.[/yellow]")
                console.print("[yellow]As a last resort, template-based generation will be used if no models can be loaded.[/yellow]")
                
                # Use our new launcher script instead of direct import
                cmd = [
                    "python", "-m", "redteamer.conversational_redteam_launcher",
                    "--target-type", target_type
                ]
            else:
                console.print("\n[bold]Running conversational red teaming in command line mode...[/bold]")
                
                # Command for CLI mode
                cmd = [
                    "python", "-m", "redteamer.conversational_redteam_cli",
                    "--target-type", target_type
                ]
            
            if model:
                cmd.extend(["--model", model])
            
            if curl_command:
                cmd.extend(["--curl-command", curl_command])
            
            cmd.extend(["--chatbot-context", context])
            
            if redteam_model:
                cmd.extend(["--redteam-model", redteam_model])
                
            cmd.extend(["--max-iterations", str(max_iterations)])
            
            if verbose:
                cmd.append("--verbose")
                
            # Run the command
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[bold red]Error launching conversational red teaming:[/bold red] {str(e)}")
    
    # Wait for user to press enter before returning to main menu
    input("\nPress Enter to return to the main menu...")
    main_menu()

def test_model_menu():
    """Menu for testing a model with a single prompt."""
    print_header()
    console.print("[bold]Test Model[/bold]")
    console.print("Send a single prompt to a model\n")
    
    # Check if ollama is available
    ollama_models = get_ollama_models()
    ollama_available = len(ollama_models) > 0
    
    # Prepare provider choices
    provider_choices = ["openai", "anthropic", "gemini"]
    if ollama_available:
        provider_choices.append("ollama")
    provider_choices.append("custom")
    
    # Ask user to select a provider using inquirer
    provider_question = [
        inquirer.List(
            "provider",
            message="Select a model provider",
            choices=provider_choices
        )
    ]
    
    provider_answer = inquirer.prompt(provider_question)
    selected_provider = provider_answer["provider"]
    
    # If custom model is selected, handle it directly
    if selected_provider == "custom":
        custom_questions = [
            inquirer.Text(
                "custom_model",
                message="Enter curl command with {prompt} placeholder",
                default="curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"llama3\",\"prompt\":\"{prompt}\"}' | jq -r '.response'"
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
        
        custom_answers = inquirer.prompt(custom_questions)
        
        # Configure the test with custom model
        curl_command = custom_answers["custom_model"]
        prompt = custom_answers["prompt"]
        system_prompt = custom_answers["system_prompt"]
        evaluate = custom_answers["evaluate"]
        verbose = custom_answers["verbose"]
        
        console.print("[yellow]Custom model testing not fully implemented in this demo[/yellow]")
        input("\nPress Enter to return to the main menu...")
        main_menu()
        return
    
    else:
        # For regular providers, check API key only for the selected provider
        # Import API key manager only when needed
        from redteamer.utils.api_key_manager import get_api_key_manager
        api_key_manager = get_api_key_manager()
        
        # Check if API key is available for non-ollama providers
        if selected_provider in ["openai", "anthropic", "gemini"]:
            api_key = api_key_manager.get_key(selected_provider)
            
            if not api_key:
                console.print(f"[bold yellow]No API key found for {selected_provider}.[/bold yellow]")
                set_key_now = inquirer.confirm(
                    f"Would you like to set an API key for {selected_provider} now?",
                    default=True
                )
                
                if set_key_now:
                    api_key = inquirer.text(f"Enter your {selected_provider.upper()} API key:")
                    api_key_manager.set_key(selected_provider, api_key)
                    console.print(f"[green]API key for {selected_provider} has been set.[/green]")
                else:
                    console.print("[yellow]Operation cancelled. API key is required.[/yellow]")
                    return main_menu()
        
        # Now fetch models for the selected provider
        console.print(f"\n[bold]Fetching models for {selected_provider}...[/bold]")
        
        # Get available models
        available_models = get_available_models(provider=selected_provider)
        
        # For ollama, use the models we fetched earlier
        if selected_provider == "ollama" and ollama_models:
            provider_models = ollama_models
        else:
            provider_models = list(available_models.get(selected_provider, {}).keys())
        
        if not provider_models:
            console.print(f"[bold red]No models available for {selected_provider}[/bold red]")
            return main_menu()
        
        # Use inquirer for model and prompt selection
        model_questions = [
            inquirer.List(
                "model",
                message=f"Select a {selected_provider} model",
                choices=provider_models
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
        
        model_answers = inquirer.prompt(model_questions)
        
        # Configure the test with selected model
        provider = selected_provider
        model = model_answers["model"]
        prompt = model_answers["prompt"]
        system_prompt = model_answers["system_prompt"]
        evaluate = model_answers["evaluate"]
        verbose = model_answers["verbose"]
    
    # Build the command
    cmd = ["python", "-m", "redteamer.cli", "test",
           "--provider", provider,
           "--model", model,
           "--prompt", prompt]
    
    # Add system prompt if provided
    if system_prompt:
        cmd.extend(["--system", system_prompt])
    
    # Add evaluator if selected
    if evaluate:
        cmd.extend(["--evaluator", "rule-based"])
    
    # Add verbose flag if selected
    if verbose:
        cmd.append("--verbose")
    
    # Run the command
    console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
    
    # Confirm before running
    confirm = inquirer.confirm("Ready to test the model?", default=True)
    if confirm:
        try:
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[bold red]Error testing model:[/bold red] {str(e)}")
    
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
    """Menu for viewing reports and results."""
    print_header()
    console.print("[bold]View Reports and Results[/bold]")
    console.print("Explore and analyze your scan results\n")
    
    # Get available results
    results = []
    if os.path.exists("results"):
        for root, dirs, files in os.walk("results"):
            for file in files:
                if file.endswith(".json"):
                    results.append(os.path.join(root, file))
    
    if not results:
        console.print("[yellow]No results found. Run a scan or evaluation first.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        main_menu()
        return
    
    # Sort results by modification time, newest first
    results = sorted(results, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Prepare display names for results (last folder + filename)
    display_results = []
    for result in results:
        parts = result.split(os.sep)
        
        # For time-based names, try to format more nicely
        filename = parts[-1]
        if "_" in filename:
            # Try to parse timestamp from filename (static_scan_20230401_123456.json)
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    scan_type = '_'.join(parts[:-2])
                    date_str = parts[-2]
                    time_str = parts[-1].split('.')[0]
                    
                    if len(date_str) == 8 and date_str.isdigit() and len(time_str) == 6 and time_str.isdigit():
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        display_name = f"{scan_type} ({formatted_date} {formatted_time})"
                    else:
                        display_name = filename
                else:
                    display_name = filename
            except:
                display_name = filename
        else:
            display_name = filename
            
        # Add parent folder for context if not in root results dir
        if len(parts) > 2:
            display_name = f"{parts[-2]}/{display_name}"
            
        display_results.append(display_name)
    
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Launch interactive results viewer",
                "Generate a new report from results",
                "Return to main menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Launch interactive results viewer":
        # Ask the user about interface preference
        console.print("\n[bold]Interface Preference[/bold]")
        console.print("  1. Streamlit UI - Provides interactive visualization with charts and filters")
        console.print("  2. Command Line - Text-based display of results")
        
        interface_idx = get_valid_index("Select interface: ", 3)
        use_streamlit = interface_idx == 1
        
        try:
            if use_streamlit:
                console.print("\n[bold]Launching Streamlit Results Viewer...[/bold]")
                console.print("[green]The results viewer will open in your browser.[/green]")
                
                # Use our new launcher script for the results viewer
                subprocess.run(["python", "-m", "redteamer.results_viewer_launcher"])
            else:
                console.print("\n[bold]Launching Command Line Results Viewer...[/bold]")
                
                # Let user select a results file
                result_question = [
                    inquirer.List(
                        "results_file",
                        message="Select results file to view",
                        choices=display_results
                    )
                ]
                
                result_answers = inquirer.prompt(result_question)
                
                # Map display name back to file path
                selected_index = display_results.index(result_answers["results_file"])
                selected_file = results[selected_index]
                
                # Run CLI results viewer command
                subprocess.run(["python", "-m", "redteamer.results_viewer_cli", selected_file])
        except Exception as e:
            if use_streamlit:
                console.print(f"[bold red]Error launching results viewer:[/bold red] {str(e)}")
            else:
                console.print(f"[bold red]Error displaying results:[/bold red] {str(e)}")
    
    elif answers["action"] == "Generate a new report from results":
        # Ask user to select a results file
        result_question = [
            inquirer.List(
                "results_file",
                message="Select results file",
                choices=display_results
            ),
            inquirer.List(
                "format",
                message="Select report format",
                choices=["markdown", "json", "csv", "pdf"]
            )
        ]
        
        result_answers = inquirer.prompt(result_question)
        
        # Map display name back to file path
        selected_index = display_results.index(result_answers["results_file"])
        selected_file = results[selected_index]
        
        # Run report generate command
        cmd = ["python", "-m", "redteamer.cli", "report", "generate",
               selected_file,
               "--format", result_answers["format"]]
        
        console.print(f"\n[bold]Running command:[/bold] {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd)
            
            # Try to get report path
            report_path = selected_file.replace(".json", f"_report.{result_answers['format']}")
            report_path = report_path.replace("results", "reports")
            
            if os.path.exists(report_path):
                console.print(f"[green]Report generated successfully:[/green] {report_path}")
        except Exception as e:
            console.print(f"[bold red]Error generating report:[/bold red] {str(e)}")
    
    elif answers["action"] == "Return to main menu":
        main_menu()
        return
    
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

def delete_results_menu():
    """Display menu for deleting scan results."""
    print_header()
    console.print("[bold]Delete Scan Results[/bold]")
    console.print("This will permanently delete scan results.\n")
    
    # Get all results
    results_dir = Path("results")
    if not results_dir.exists():
        console.print("[yellow]No results directory found.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        return main_menu()
    
    # Get all result files
    result_files = []
    
    # Get static scan results
    static_results = list(results_dir.glob("static_scan_*.json"))
    for result in static_results:
        result_files.append(("Static", result))
    
    # Get conversational scan results
    convo_results = list(results_dir.glob("conversational_*.json"))
    for result in convo_results:
        result_files.append(("Conversation", result))
    
    # Check reports directory
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.md")) + list(reports_dir.glob("*.pdf")) + list(reports_dir.glob("*.json"))
        for report in report_files:
            result_files.append(("Report", report))
    
    if not result_files:
        console.print("[yellow]No result files found.[/yellow]")
        input("\nPress Enter to return to the main menu...")
        return main_menu()
    
    # Display results
    console.print("[bold]Available Results:[/bold]")
    
    table = Table()
    table.add_column("#", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Filename", style="green")
    table.add_column("Date", style="yellow")
    table.add_column("Size", style="blue")
    
    for i, (result_type, result_path) in enumerate(result_files):
        # Get file stats
        stats = result_path.stat()
        # Format date
        date = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
        # Format size
        if stats.st_size < 1024:
            size = f"{stats.st_size} B"
        elif stats.st_size < 1024 * 1024:
            size = f"{stats.st_size / 1024:.1f} KB"
        else:
            size = f"{stats.st_size / (1024 * 1024):.1f} MB"
        
        table.add_row(
            str(i+1),
            result_type,
            result_path.name,
            date,
            size
        )
    
    console.print(table)
    console.print()
    
    # Options
    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Delete specific result",
                "Delete all results",
                "Return to main menu"
            ]
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers["action"] == "Delete specific result":
        # Ask which result to delete
        idx = get_valid_index("Enter the number of the result to delete: ", len(result_files) + 1)
        result_type, result_path = result_files[idx]
        
        # Confirm deletion
        confirm = Confirm.ask(f"Are you sure you want to delete {result_path.name}?")
        if confirm:
            try:
                result_path.unlink()
                console.print(f"[green]✓[/green] Deleted {result_path.name}")
            except Exception as e:
                console.print(f"[red]Error deleting file:[/red] {str(e)}")
        else:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            
    elif answers["action"] == "Delete all results":
        # Confirm deletion
        confirm = Confirm.ask(f"Are you sure you want to delete ALL {len(result_files)} results? This cannot be undone.")
        if confirm:
            # Double confirm
            double_confirm = Confirm.ask("Really delete all results?")
            if double_confirm:
                errors = 0
                for _, result_path in result_files:
                    try:
                        result_path.unlink()
                    except Exception:
                        errors += 1
                
                if errors == 0:
                    console.print(f"[green]✓[/green] Deleted all {len(result_files)} results.")
                else:
                    console.print(f"[yellow]Deleted {len(result_files) - errors} results. {errors} errors occurred.[/yellow]")
            else:
                console.print("[yellow]Deletion cancelled.[/yellow]")
        else:
            console.print("[yellow]Deletion cancelled.[/yellow]")
    
    # Return to main menu
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