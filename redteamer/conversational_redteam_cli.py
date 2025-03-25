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
        console.print(f"[bold]Target:[/bold] {args.target_type}/{args.model}")
    
    console.print(f"[bold]Context:[/bold] {args.chatbot_context}")
    
    if args.system_prompt:
        console.print(f"[bold]System prompt:[/bold] {args.system_prompt}")
    
    if args.redteam_model_id:
        console.print(f"[bold]Red team model:[/bold] {args.redteam_model_id}")
    else:
        console.print("[bold]Red team model:[/bold] (using default)")
    
    console.print(f"[bold]Maximum iterations:[/bold] {args.max_iterations}")
    console.print()
    
    # Prepare model configuration
    model_config = create_model_config(args)
    
    # Create the conversational red team instance
    console.print("[bold]Initializing conversational red teaming...[/bold]")
    redteam = ConversationalRedTeam(
        logging_level="DEBUG" if args.verbose else "INFO",
        chatbot_context=args.chatbot_context,
        redteam_model_id=args.redteam_model_id,
        max_iterations=args.max_iterations
    )
    
    # Register CLI callbacks for progress updates
    register_cli_callbacks(redteam)
    
    # Run the conversational red teaming
    console.print("\n[bold]Starting conversational red teaming...[/bold]")
    console.print("[italic]This will test how the model responds to adversarial prompts in a conversation.[/italic]\n")
    
    try:
        # Run the red team conversation
        results = await redteam.run_redteam_conversation(model_config)
        
        # Save results
        results_path = os.path.join("chatbot_evals", f"redteam_conversation_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        display_conversation_summary(results)
        
        console.print(f"\n[bold green]✓ Conversational red teaming completed![/bold green]")
        console.print(f"[bold]Results saved to:[/bold] {results_path}")
        console.print("[dim]Use the 'results viewer' to analyze these results in detail.[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error during conversational red teaming:[/bold red] {str(e)}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())

def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration from arguments."""
    if args.curl_command:
        # Custom model
        return {
            "provider": "custom",
            "model_id": "custom-model",
            "curl_command": args.curl_command
        }
    else:
        # Standard provider model
        config = {
            "provider": args.target_type,
            "model_id": args.model,
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        # Add system prompt if provided
        if args.system_prompt:
            config["system_prompt"] = args.system_prompt
        
        return config

def register_cli_callbacks(redteam):
    """Register callbacks for CLI progress updates."""
    
    # Callback for new iteration
    def on_iteration_start(iteration: int, max_iterations: int, prompt: str):
        console.print(f"\n[bold][Iteration {iteration}/{max_iterations}][/bold]")
        console.print(Panel(prompt, title="Red Team Prompt", border_style="red"))
    
    # Callback for model response
    def on_model_response(response: str):
        console.print(Panel(response, title="Model Response", border_style="blue"))
    
    # Callback for evaluation
    def on_evaluation(evaluation_result: Dict[str, Any]):
        success = evaluation_result.get("success", False)
        color = "red" if success else "green"
        icon = "✗" if success else "✓"
        
        console.print(f"[bold][Evaluation][/bold] [{color}]{icon} {'Prompt succeeded' if success else 'Model resisted attack'}[/{color}]")
        
        if "explanation" in evaluation_result:
            console.print(f"[dim]{evaluation_result['explanation']}[/dim]")
        
        if "matched_patterns" in evaluation_result:
            patterns = evaluation_result["matched_patterns"]
            if patterns:
                console.print("[bold]Matched patterns:[/bold]")
                for pattern in patterns:
                    console.print(f"- {pattern}")
    
    # Callback for error
    def on_error(error_message: str):
        console.print(f"[bold red]Error:[/bold red] {error_message}")
    
    # Register callbacks
    redteam.set_callback("iteration_start", on_iteration_start)
    redteam.set_callback("model_response", on_model_response)
    redteam.set_callback("evaluation", on_evaluation)
    redteam.set_callback("error", on_error)

def display_conversation_summary(results: Dict[str, Any]):
    """Display a summary of the conversational red teaming results."""
    console.print("\n[bold]Conversational Red Team Results Summary[/bold]")
    
    # Extract metrics
    total_iterations = results.get("total_iterations", 0)
    successful_iterations = sum(1 for turn in results.get("conversation", [])
                              if turn.get("evaluation", {}).get("success", False))
    vulnerability_rate = 0
    
    if total_iterations > 0:
        vulnerability_rate = (successful_iterations / total_iterations) * 100
    
    # Get model info
    model_info = results.get("model_info", {})
    model_name = f"{model_info.get('provider', 'unknown')}/{model_info.get('model_id', 'unknown')}"
    
    # Display summary
    console.print(f"[bold]Model tested:[/bold] {model_name}")
    console.print(f"[bold]Total conversation turns:[/bold] {total_iterations}")
    console.print(f"[bold]Successful attacks:[/bold] {successful_iterations}")
    console.print(f"[bold]Vulnerability rate:[/bold] {vulnerability_rate:.2f}%")
    
    # Display techniques that worked
    if "conversation" in results:
        attack_types = {}
        for turn in results["conversation"]:
            evaluation = turn.get("evaluation", {})
            if evaluation.get("success", False):
                attack_type = turn.get("metadata", {}).get("technique", "unknown")
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        if attack_types:
            console.print("\n[bold]Successful attack techniques:[/bold]")
            sorted_types = sorted(attack_types.items(), key=lambda x: x[1], reverse=True)
            for attack_type, count in sorted_types:
                console.print(f"- {attack_type}: {count} successful attacks")

if __name__ == "__main__":
    main() 