#!/usr/bin/env python3
"""
Simple CLI tool to test the model selection functionality.
Run this to try the dynamic model selection feature.
"""

import os
import json
from rich.console import Console
from redteamer.models import get_all_available_models, select_model_interactively

console = Console()

def main():
    """Run model selection test"""
    console.print("[bold]RedTeamer Dynamic Model Selection Test[/bold]")
    console.print("This tool tests the ability to fetch available AI models from different providers.")
    
    # Get available providers
    console.print("\n[bold]Available model providers:[/bold]")
    provider_options = ["openai", "anthropic", "gemini", "ollama", "custom"]
    
    for i, provider in enumerate(provider_options):
        console.print(f"{i+1}. {provider}")
    
    # Let the user select a provider first
    provider_choice = input("\nSelect a provider (1-5): ")
    try:
        provider_idx = int(provider_choice) - 1
        if 0 <= provider_idx < len(provider_options):
            selected_provider = provider_options[provider_idx]
        else:
            console.print("[yellow]Invalid selection. Using 'openai' as default.[/yellow]")
            selected_provider = "openai"
    except ValueError:
        console.print("[yellow]Invalid input. Using 'openai' as default.[/yellow]")
        selected_provider = "openai"
    
    # Now fetch models only for the selected provider
    console.print(f"\n[bold]Fetching available models for {selected_provider}...[/bold]")
    
    if selected_provider == "custom":
        console.print("[bold cyan]Custom model selected[/bold cyan]")
        custom_model_name = input("Enter a name for your custom model: ")
        custom_model_cmd = input("Enter your custom model command (use {prompt} as placeholder): ")
        
        console.print(f"\n[bold green]Custom model '{custom_model_name}' configured[/bold green]")
        return
    
    # Get models for the selected provider
    available_models = get_all_available_models(provider=selected_provider)
    
    # Print available models for the selected provider
    console.print(f"\n[bold]Available {selected_provider} models:[/bold]")
    provider_models = available_models.get(selected_provider, {})
    
    if not provider_models:
        console.print(f"[yellow]No models available for {selected_provider}[/yellow]")
        return
    
    model_count = len(provider_models)
    console.print(f"[bold cyan]{selected_provider}[/bold cyan] ({model_count} models):")
    
    # Print up to 5 models, then summarize the rest
    model_names = list(provider_models.keys())
    if len(model_names) <= 5:
        console.print(f"   {', '.join(model_names)}")
    else:
        console.print(f"   {', '.join(model_names[:5])} and {len(model_names) - 5} more...")
    
    # Interactive selection of model
    console.print("\n[bold]Select a model:[/bold]")
    for i, model_name in enumerate(model_names):
        console.print(f"{i+1}. {model_name}")
    
    model_choice = input(f"\nSelect a model (1-{len(model_names)}): ")
    try:
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(model_names):
            selected_model = model_names[model_idx]
        else:
            selected_model = model_names[0]
            console.print(f"[yellow]Invalid selection. Using '{selected_model}' as default.[/yellow]")
    except ValueError:
        selected_model = model_names[0]
        console.print(f"[yellow]Invalid input. Using '{selected_model}' as default.[/yellow]")
    
    selected_model_str = f"{selected_provider}:{selected_model}"
    console.print(f"\n[bold green]Selected model: {selected_model_str}[/bold green]")
    
    # Show detailed info about the selected model
    provider, model_id = selected_model_str.split(":", 1)
    model_config = provider_models.get(model_id, {})
    
    console.print("\n[bold]Model configuration:[/bold]")
    for key, value in model_config.items():
        console.print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 