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
    
    console.print("\n[bold]Fetching available models...[/bold]")
    
    # Get all available models
    available_models = get_all_available_models()
    
    # Print available models by provider
    console.print("\n[bold]Available models by provider:[/bold]")
    for provider, models in available_models.items():
        model_count = len(models)
        console.print(f"[bold cyan]{provider}[/bold cyan] ({model_count} models):")
        
        # Print up to 5 models, then summarize the rest
        model_names = list(models.keys())
        if len(model_names) <= 5:
            console.print(f"   {', '.join(model_names)}")
        else:
            console.print(f"   {', '.join(model_names[:5])} and {len(model_names) - 5} more...")
    
    # Interactive selection
    console.print("\n[bold]Interactive Model Selection:[/bold]")
    selected_model = select_model_interactively()
    
    console.print(f"\n[bold green]Selected model: {selected_model}[/bold green]")
    
    # Show detailed info about the selected model
    provider, model_id = selected_model.split(":", 1)
    if provider in available_models and model_id in available_models[provider]:
        model_config = available_models[provider][model_id]
        console.print("\n[bold]Model configuration:[/bold]")
        console.print(f"Temperature: {model_config.get('temperature', 0.7)}")
        console.print(f"Max tokens: {model_config.get('max_tokens', 1000)}")
        
        api_key_env = model_config.get('api_key_env')
        if api_key_env:
            if api_key_env in os.environ:
                console.print(f"API key status: [green]Available[/green] (via {api_key_env})")
            else:
                console.print(f"API key status: [yellow]Missing[/yellow] (needs {api_key_env})")
        else:
            console.print("API key status: [green]Not required[/green]")
    
    console.print("\n[bold]Test complete![/bold]")

if __name__ == "__main__":
    main() 