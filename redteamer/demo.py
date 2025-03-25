#!/usr/bin/env python3
"""
Demo script to run a static scan without checking API keys at startup.
"""

import os
import sys
import subprocess
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

def main():
    """Run a demo of the static scan without checking API keys at startup."""
    console.print("[bold]RedTeamer Static Scan Demo[/bold]")
    console.print("This demo shows how to run a static scan without checking API keys at startup.")
    
    # Define available providers
    providers = ["openai", "anthropic", "gemini", "ollama", "custom"]
    
    # Check if ollama is available
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ollama_available = True
    except:
        ollama_available = False
        # Remove ollama from providers if not available
        providers.remove("ollama")
    
    # Show provider selection
    console.print("\n[bold]Available Model Providers:[/bold]")
    for i, provider in enumerate(providers):
        console.print(f"{i+1}. {provider}")
    
    # Let user choose a provider
    provider_idx = Prompt.ask(
        "Select a provider",
        choices=[str(i+1) for i in range(len(providers))],
        default="1"
    )
    
    try:
        provider_choice = providers[int(provider_idx) - 1]
    except (ValueError, IndexError):
        console.print("[bold red]Invalid selection[/bold red]")
        return
    
    # Handle custom provider
    if provider_choice == "custom":
        custom_model_name = Prompt.ask("Enter a name for your custom model", default="custom-model")
        custom_model = Prompt.ask(
            "Enter your custom model curl command (use {prompt} as placeholder)",
            default="curl -X POST http://localhost:8080/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"{prompt}\"}'"
        )
        
        if "{prompt}" not in custom_model:
            console.print("[bold red]Error: Custom model command must contain {prompt} placeholder.[/bold red]")
            return
            
        # Ask for number of prompts
        num_prompts = Prompt.ask(
            "Number of adversarial prompts to generate", 
            default="10"
        )
        
        if Confirm.ask(f"Run static scan with custom model '{custom_model_name}'?"):
            # Launch static scan with custom model
            cmd = [
                "python", "-m", "redteamer.static_scan_launcher",
                "--custom-model", custom_model,
                "--custom-model-name", custom_model_name,
                "--num-prompts", num_prompts
            ]
            
            console.print(f"\n[bold]Running:[/bold] {' '.join(cmd)}")
            subprocess.run(cmd)
    else:
        # For providers that need API keys, check the key now
        if provider_choice in ["openai", "anthropic", "gemini"]:
            from redteamer.utils.api_key_manager import get_api_key_manager
            api_key_manager = get_api_key_manager()
            
            api_key = api_key_manager.get_key(provider_choice)
            
            if not api_key:
                console.print(f"[bold yellow]No API key found for {provider_choice}.[/bold yellow]")
                
                if Confirm.ask(f"Would you like to set an API key for {provider_choice} now?"):
                    api_key = Prompt.ask(f"Enter your {provider_choice.upper()} API key")
                    api_key_manager.set_key(provider_choice, api_key)
                    console.print(f"[green]API key for {provider_choice} has been set.[/green]")
                else:
                    console.print("[yellow]Operation cancelled. API key is required.[/yellow]")
                    return
        
        # Now fetch models for the selected provider
        console.print(f"\n[bold]Fetching models for {provider_choice}...[/bold]")
        
        # Import models module only when needed
        from redteamer.models import get_all_available_models
        provider_models = get_all_available_models(provider=provider_choice).get(provider_choice, {})
        
        if not provider_models:
            console.print(f"[bold red]No models available for {provider_choice}[/bold red]")
            return
        
        # Show model selection
        console.print(f"\n[bold]Available {provider_choice} models:[/bold]")
        model_names = list(provider_models.keys())
        
        for i, model_name in enumerate(model_names):
            console.print(f"{i+1}. {model_name}")
        
        model_idx = Prompt.ask(
            "Select a model",
            choices=[str(i+1) for i in range(len(model_names))],
            default="1"
        )
        
        try:
            model_choice = model_names[int(model_idx) - 1]
        except (ValueError, IndexError):
            console.print("[bold red]Invalid model selection[/bold red]")
            return
            
        # Ask for number of prompts
        num_prompts = Prompt.ask(
            "Number of adversarial prompts to generate", 
            default="10"
        )
        
        if Confirm.ask(f"Run static scan with {provider_choice}/{model_choice}?"):
            # Launch static scan with selected model
            cmd = [
                "python", "-m", "redteamer.static_scan_launcher",
                "--provider", provider_choice,
                "--model", model_choice,
                "--num-prompts", num_prompts
            ]
            
            console.print(f"\n[bold]Running:[/bold] {' '.join(cmd)}")
            subprocess.run(cmd)

if __name__ == "__main__":
    main() 