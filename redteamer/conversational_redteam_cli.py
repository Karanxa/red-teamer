#!/usr/bin/env python
"""
Command-line launcher for conversational red teaming.

This script provides a text-based interface for running conversational
red teaming evaluations without requiring Streamlit.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from typing import Dict, Any, Optional

# Make this script runnable from any location
script_dir = Path(__file__).parent
os.chdir(script_dir.parent)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversational_redteam_cli")

# Initialize rich console
console = Console()

def main():
    """Run the conversational red teaming application in CLI mode."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational LLM Red Teaming (CLI Mode)")
    parser.add_argument("--target-type", default="ollama", help="Target model type (ollama, openai, anthropic, etc.)")
    parser.add_argument("--model", default="llama3", help="Model name")
    parser.add_argument("--system-prompt", help="System prompt for the target model")
    parser.add_argument("--chatbot-context", default="You are a helpful AI assistant.", help="Description of the chatbot being tested")
    parser.add_argument("--redteam-model-id", help="Model ID for red teaming (if not provided, will use default)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of conversation iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--curl-command", help="Custom curl command for target model")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", 
                      help="Device to run the model on: 'cpu' (slower but works everywhere) or 'gpu' (faster but requires compatible hardware)")
    parser.add_argument("--fallback-mode", action="store_true", 
                      help="Use template-based generation only, without attempting to load any models (useful when dependencies have issues)")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Run the conversational red team
    try:
        asyncio.run(run_conversational_redteam_cli(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error running conversational red team:[/bold red] {str(e)}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())

async def run_conversational_redteam_cli(args):
    """
    Run the conversational red teaming process with CLI output.
    
    Args:
        args: Command line arguments
    """
    console = Console()
    
    console.print("[bold blue]Conversational LLM Red Teaming[/bold blue]")
    console.print("Running evaluation in CLI mode\n")
    
    # Ensure required directories exist
    for directory in ["results", "reports", "chatbot_evals"]:
        os.makedirs(directory, exist_ok=True)
    
    # Generate a timestamp for this scan
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    created_at = datetime.now().isoformat()
    
    # Import the ConversationalRedTeam class
    try:
        from redteamer.red_team.conversational_redteam import ConversationalRedTeam
    except ImportError as e:
        console.print("[bold red]Error:[/bold red] Failed to import ConversationalRedTeam class.")
        console.print(f"[red]{str(e)}[/red]")
        return
    
    # Show configuration
    console.print("[bold]Conversational Red Team Configuration:[/bold]")
    
    if args.curl_command:
        console.print(f"[bold]Target:[/bold] Custom model via curl")
        if args.verbose:
            console.print(f"[dim]Command: {args.curl_command}[/dim]")
    else:
        console.print(f"[bold]Target:[/bold] {args.target_type}")
    
    console.print(f"[bold]Context:[/bold] {args.chatbot_context}")
    
    if hasattr(args, 'system_prompt') and args.system_prompt:
        console.print(f"[bold]System prompt:[/bold] {args.system_prompt}")
    
    if hasattr(args, 'redteam_model_id') and args.redteam_model_id:
        console.print(f"[bold]Red team model:[/bold] {args.redteam_model_id}")
    elif hasattr(args, 'model') and args.model:
        console.print(f"[bold]Red team model:[/bold] {args.model}")
    else:
        console.print("[bold]Red team model:[/bold] (using default)")
    
    console.print(f"[bold]Maximum iterations:[/bold] {args.max_iterations}")
    console.print()
    
    # Show device selection
    if hasattr(args, 'device'):
        if args.device == "cpu":
            console.print(f"[bold]Device:[/bold] CPU (optimized for compatibility)")
        else:
            console.print(f"[bold]Device:[/bold] GPU (optimized for speed)")
    
    # Show fallback mode information
    if hasattr(args, 'fallback_mode') and args.fallback_mode:
        console.print("[bold yellow]Running in fallback mode - using templated prompts only[/bold yellow]")
        console.print("This mode bypasses model loading entirely and uses pre-defined adversarial prompt templates.")
    console.print()
    
    # Create model configuration
    model_config = create_model_config(args)
    
    # Create the conversational red team instance
    console.print("[bold]Initializing conversational red teaming...[/bold]")
    redteam = ConversationalRedTeam(
        target_model_type=args.target_type,
        chatbot_context=args.chatbot_context,
        redteam_model_id=getattr(args, 'redteam_model_id', getattr(args, 'model', None)),
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        quant_mode="cpu" if getattr(args, 'device', '') == "cpu" else "auto",
        fallback_mode=getattr(args, 'fallback_mode', False),
        device=getattr(args, 'device', 'gpu'),
        curl_command=getattr(args, 'curl_command', None)
    )
    
    # Force template mode if fallback mode is enabled
    if getattr(args, 'fallback_mode', False):
        redteam.using_templates = True
        redteam.using_fallback = True
        console.print("[yellow]Using pre-defined templates for adversarial prompts - no model loading required[/yellow]")
    
    # Register CLI callbacks for progress updates
    def cli_progress_callback(iteration, total, exchange, vulnerabilities_count=0):
        """CLI callback for progress updates"""
        found_vulnerability = exchange.get('found_vulnerability', False)
        severity = exchange.get('severity', 'none')
        
        if found_vulnerability:
            console.print(f"\n[bold red]Found potential vulnerability in iteration {iteration}![/bold red]")
            console.print(f"Severity: {severity.upper()}")
            console.print(f"Type: {exchange.get('vulnerability_type', 'Unknown')}")
            console.print(f"Total vulnerabilities found: {vulnerabilities_count}")
        else:
            console.print(f"\nIteration {iteration}/{total} completed.")
            
        console.print(f"\n[bold green]User/Adversarial Prompt:[/bold green]")
        console.print(exchange.get('prompt', 'No prompt'))
        
        console.print(f"\n[bold blue]Target Model Response:[/bold blue]")
        console.print(exchange.get('response', 'No response'))
        
        console.print("\n" + "-" * 80 + "\n")
    
    try:
        # Run the red team conversation
        results = await redteam.run_redteam_conversation(model_config, cli_progress_callback)
        
        # Save results
        results_path = os.path.join("chatbot_evals", f"redteam_conversation_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        console.print("\n[bold]Red Teaming Complete![/bold]")
        console.print(f"Ran {results.get('num_iterations', 0)} iterations")
        
        vulnerabilities = results.get('vulnerabilities', [])
        if vulnerabilities:
            console.print(f"[bold red]Found {len(vulnerabilities)} potential vulnerabilities![/bold red]")
            for idx, vuln in enumerate(vulnerabilities):
                console.print(f"\n[bold]Vulnerability {idx+1}:[/bold]")
                console.print(f"Type: {vuln.get('vulnerability_type', 'Unknown')}")
                console.print(f"Severity: {vuln.get('severity', 'unknown').upper()}")
                console.print(f"Prompt: {vuln.get('prompt', 'No prompt')[:200]}...")
                console.print(f"Response: {vuln.get('response', 'No response')[:200]}...")
        else:
            console.print("[bold green]No vulnerabilities detected.[/bold green]")
            
        console.print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error during red teaming:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration dictionary from args."""
    model_config = {}
    model_config['model'] = args.target_type
    
    if hasattr(args, 'system_prompt') and args.system_prompt:
        model_config['system'] = args.system_prompt
    else:
        model_config['system'] = ""
        
    if hasattr(args, 'curl_command') and args.curl_command:
        model_config['curl_command'] = args.curl_command
    
    return model_config

if __name__ == "__main__":
    main() 