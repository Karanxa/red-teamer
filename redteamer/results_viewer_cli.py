#!/usr/bin/env python
"""
Command-line results viewer for Red Teaming Framework.

This script provides a text-based interface for viewing red teaming results
without requiring Streamlit.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("results_viewer_cli")

# Initialize rich console
console = Console()

def main():
    """Run the results viewer application in CLI mode."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Red Teaming Results Viewer (CLI Mode)")
    parser.add_argument("results_file", nargs="?", help="Path to results file to view")
    parser.add_argument("--list", action="store_true", help="List available results files")
    parser.add_argument("--output-dir", default="results", help="Directory containing results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # If only --list is specified, show available results
        if args.list or not args.results_file:
            list_results_files(args.output_dir)
            if not args.results_file:
                return
        
        # If results file is specified, display it
        display_results_cli(args.results_file, args.verbose)
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error viewing results:[/bold red] {str(e)}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())

def list_results_files(output_dir="results"):
    """List available results files."""
    console.print("[bold blue]Available Results Files[/bold blue]\n")
    
    # Get available results
    results = []
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".json"):
                    results.append(os.path.join(root, file))
    
    if not results:
        console.print("[yellow]No results found. Run a scan or evaluation first.[/yellow]")
        return
    
    # Sort results by modification time, newest first
    results = sorted(results, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Display available results
    table = Table(title="Results Files", box=box.ROUNDED)
    table.add_column("#", style="cyan")
    table.add_column("Filename", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Date", style="yellow")
    table.add_column("Path", style="dim")
    
    for i, result_path in enumerate(results):
        path_parts = result_path.split(os.sep)
        filename = path_parts[-1]
        
        # Try to determine type from filename
        result_type = "Unknown"
        if "static_scan" in filename:
            result_type = "Static Scan"
        elif "redteam_conversation" in filename:
            result_type = "Conversation"
        elif "redteam" in filename:
            result_type = "Red Team"
        
        # Format date from file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(result_path))
        date_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
        
        table.add_row(
            str(i + 1),
            filename,
            result_type,
            date_str,
            result_path
        )
    
    console.print(table)
    console.print("\n[dim]Use 'python -m redteamer.results_viewer_cli [path]' to view a specific file[/dim]")

def display_results_cli(results_file, verbose=False):
    """
    Display results in CLI mode.
    
    Args:
        results_file: Path to results file
        verbose: Whether to show verbose output
    """
    try:
        # Load results file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Determine result type
        result_type = determine_result_type(results)
        
        console.print(f"[bold blue]Red Teaming Results Viewer[/bold blue] - {result_type}")
        console.print(f"[dim]File: {results_file}[/dim]\n")
        
        # Display results based on type
        if result_type == "Static Scan":
            display_static_results(results, verbose)
        elif result_type == "Conversational Red Team":
            display_conversation_results(results, verbose)
        elif result_type == "Standard Red Team":
            display_standard_results(results, verbose)
        else:
            console.print("[yellow]Unknown result type. Displaying general information.[/yellow]")
            display_general_results(results, verbose)
            
    except Exception as e:
        console.print(f"[bold red]Error reading results file:[/bold red] {str(e)}")
        raise

def determine_result_type(results):
    """Determine the type of results data."""
    if "conversation" in results:
        return "Conversational Red Team"
    elif "total_vectors" in results and "model_results" in results:
        return "Static Scan"
    elif "benchmark_results" in results:
        return "Standard Red Team"
    else:
        return "Unknown Result Type"

def display_static_results(results, verbose=False):
    """Display results from a static scan."""
    # Extract basic metrics
    total_vectors = results.get("total_vectors", 0)
    successful_attacks = 0
    vulnerability_rate = 0
    models_tested = []
    
    # Process model results
    model_results = results.get("model_results", [])
    for model_result in model_results:
        model_name = f"{model_result.get('provider', 'unknown')}/{model_result.get('model_id', 'unknown')}"
        models_tested.append(model_name)
        
        # Count successful attacks
        vector_results = model_result.get("vector_results", [])
        for vector_result in vector_results:
            if vector_result.get("success", False):
                successful_attacks += 1
    
    # Calculate vulnerability rate
    if total_vectors > 0 and len(model_results) > 0:
        vulnerability_rate = (successful_attacks / (total_vectors * len(model_results))) * 100
    
    # Display summary
    console.print("[bold]Scan Summary[/bold]")
    console.print(f"Models tested: {', '.join(models_tested)}")
    console.print(f"Total attack vectors: {total_vectors}")
    console.print(f"Successful attacks: {successful_attacks}")
    console.print(f"Vulnerability rate: {vulnerability_rate:.2f}%")
    
    # Display model summaries
    console.print("\n[bold]Model Summaries[/bold]")
    for model_result in model_results:
        model_name = f"{model_result.get('provider', 'unknown')}/{model_result.get('model_id', 'unknown')}"
        model_vectors = model_result.get("vector_results", [])
        model_successes = sum(1 for v in model_vectors if v.get("success", False))
        model_rate = (model_successes / len(model_vectors)) * 100 if model_vectors else 0
        
        console.print(f"\n[bold]{model_name}[/bold]")
        console.print(f"Vectors tested: {len(model_vectors)}")
        console.print(f"Successful attacks: {model_successes}")
        console.print(f"Vulnerability rate: {model_rate:.2f}%")
        
        # Display top categories if available
        categories = {}
        for vector_result in model_vectors:
            if vector_result.get("success", False):
                category = vector_result.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
        
        if categories:
            console.print("\nTop vulnerable categories:")
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories[:3]:  # Show top 3
                console.print(f"- {category}: {count} successful attacks")
    
    # Display vector details if verbose
    if verbose:
        console.print("\n[bold]Attack Vector Details[/bold]")
        
        for model_result in model_results:
            model_name = f"{model_result.get('provider', 'unknown')}/{model_result.get('model_id', 'unknown')}"
            vector_results = model_result.get("vector_results", [])
            
            console.print(f"\n[bold]Vectors for {model_name}:[/bold]")
            
            # Create a table for vector results
            table = Table(box=box.SIMPLE)
            table.add_column("ID", style="cyan")
            table.add_column("Success", style="green")
            table.add_column("Category", style="magenta")
            table.add_column("Severity", style="yellow")
            
            for vector_result in vector_results:
                vector_id = vector_result.get("id", "unknown")
                success = vector_result.get("success", False)
                category = vector_result.get("category", "unknown")
                severity = vector_result.get("severity", "unknown")
                
                success_str = "[red]✓[/red]" if success else "[green]✗[/green]"
                
                table.add_row(
                    vector_id,
                    success_str,
                    category,
                    severity
                )
            
            console.print(table)
            
            # Ask if user wants to see detailed vector information
            console.print("\nTo see details of a specific vector, enter its ID (or press Enter to continue):")
            vector_id = input()
            
            if vector_id:
                # Find the vector with the specified ID
                target_vector = next((v for v in vector_results if v.get("id") == vector_id), None)
                
                if target_vector:
                    display_vector_detail(target_vector)
                else:
                    console.print(f"[yellow]Vector with ID '{vector_id}' not found.[/yellow]")

def display_vector_detail(vector):
    """Display detailed information about a specific vector."""
    console.print("\n[bold]Vector Details[/bold]")
    
    # Create a panel for the prompt
    prompt = vector.get("prompt", "No prompt available")
    console.print(Panel(prompt, title="Prompt", border_style="cyan"))
    
    # Display system prompt if available
    if "system_prompt" in vector and vector["system_prompt"]:
        console.print(Panel(vector["system_prompt"], title="System Prompt", border_style="cyan"))
    
    # Display response if available
    if "response" in vector and vector["response"]:
        console.print(Panel(vector["response"], title="Model Response", border_style="blue"))
    
    # Display evaluation information
    console.print("\n[bold]Evaluation[/bold]")
    success = vector.get("success", False)
    success_str = "[red]Yes[/red]" if success else "[green]No[/green]"
    console.print(f"Attack succeeded: {success_str}")
    
    if "explanation" in vector:
        console.print(f"Explanation: {vector['explanation']}")
    
    if "matched_patterns" in vector and vector["matched_patterns"]:
        console.print("\nMatched patterns:")
        for pattern in vector["matched_patterns"]:
            console.print(f"- {pattern}")
    
    # Display metadata
    if "metadata" in vector and vector["metadata"]:
        console.print("\n[bold]Metadata[/bold]")
        for key, value in vector["metadata"].items():
            console.print(f"{key}: {value}")

def display_conversation_results(results, verbose=False):
    """Display results from a conversational red team evaluation."""
    # Extract basic information
    model_info = results.get("model_info", {})
    model_name = f"{model_info.get('provider', 'unknown')}/{model_info.get('model_id', 'unknown')}"
    conversation = results.get("conversation", [])
    
    # Calculate metrics
    total_turns = len(conversation)
    successful_turns = sum(1 for turn in conversation if turn.get("evaluation", {}).get("success", False))
    vulnerability_rate = (successful_turns / total_turns) * 100 if total_turns > 0 else 0
    
    # Display summary
    console.print("[bold]Conversation Summary[/bold]")
    console.print(f"Model tested: {model_name}")
    console.print(f"Total conversation turns: {total_turns}")
    console.print(f"Successful attacks: {successful_turns}")
    console.print(f"Vulnerability rate: {vulnerability_rate:.2f}%")
    
    # Display techniques that worked
    attack_types = {}
    for turn in conversation:
        evaluation = turn.get("evaluation", {})
        if evaluation.get("success", False):
            attack_type = turn.get("metadata", {}).get("technique", "unknown")
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
    
    if attack_types:
        console.print("\n[bold]Successful attack techniques:[/bold]")
        sorted_types = sorted(attack_types.items(), key=lambda x: x[1], reverse=True)
        for attack_type, count in sorted_types:
            console.print(f"- {attack_type}: {count} successful attacks")
    
    # Show the conversation if verbose
    if verbose:
        console.print("\n[bold]Conversation[/bold]")
        
        for i, turn in enumerate(conversation):
            # Get turn information
            prompt = turn.get("prompt", "No prompt available")
            response = turn.get("response", "No response available")
            evaluation = turn.get("evaluation", {})
            success = evaluation.get("success", False)
            
            # Display prompt and response
            console.print(f"\n[bold]Turn {i+1}/{total_turns}[/bold]")
            console.print(Panel(prompt, title="Red Team Prompt", border_style="red"))
            console.print(Panel(response, title="Model Response", border_style="blue"))
            
            # Display evaluation
            success_str = "[red]Succeeded[/red]" if success else "[green]Failed[/green]"
            console.print(f"Attack {success_str}")
            
            if "explanation" in evaluation:
                console.print(f"Explanation: {evaluation['explanation']}")
            
            if "matched_patterns" in evaluation and evaluation["matched_patterns"]:
                console.print("\nMatched patterns:")
                for pattern in evaluation["matched_patterns"]:
                    console.print(f"- {pattern}")
            
            # Prompt to continue
            if i < total_turns - 1:
                console.print("\nPress Enter to see next turn (or 'q' to quit):")
                user_input = input()
                if user_input.lower() == 'q':
                    break

def display_standard_results(results, verbose=False):
    """Display results from a standard red team evaluation."""
    # Extract benchmark information
    benchmark_info = results.get("benchmark_info", {})
    benchmark_name = benchmark_info.get("name", "Unknown Benchmark")
    benchmark_desc = benchmark_info.get("description", "No description available")
    
    # Extract results
    benchmark_results = results.get("benchmark_results", [])
    models_tested = [f"{r.get('provider', 'unknown')}/{r.get('model_id', 'unknown')}" for r in benchmark_results]
    
    # Display summary
    console.print(f"[bold]{benchmark_name}[/bold]")
    console.print(benchmark_desc)
    console.print(f"\nModels tested: {', '.join(models_tested)}")
    
    # Display results for each model
    for model_result in benchmark_results:
        model_name = f"{model_result.get('provider', 'unknown')}/{model_result.get('model_id', 'unknown')}"
        vectors = model_result.get("vector_results", [])
        total = len(vectors)
        successful = sum(1 for v in vectors if v.get("success", False))
        rate = (successful / total) * 100 if total > 0 else 0
        
        console.print(f"\n[bold]Results for {model_name}[/bold]")
        console.print(f"Vectors tested: {total}")
        console.print(f"Successful attacks: {successful}")
        console.print(f"Vulnerability rate: {rate:.2f}%")
        
        # Display vector table if verbose
        if verbose:
            table = Table(box=box.SIMPLE)
            table.add_column("ID", style="cyan")
            table.add_column("Category", style="magenta")
            table.add_column("Success", style="green")
            table.add_column("Severity", style="yellow")
            
            for vector in vectors:
                vector_id = vector.get("id", "unknown")
                category = vector.get("category", "unknown")
                success = vector.get("success", False)
                severity = vector.get("severity", "unknown")
                
                success_str = "[red]✓[/red]" if success else "[green]✗[/green]"
                
                table.add_row(
                    vector_id,
                    category,
                    success_str,
                    severity
                )
            
            console.print(table)
            
            # Prompt to view vector details
            console.print("\nTo see details of a specific vector, enter its ID (or press Enter to continue):")
            vector_id = input()
            
            if vector_id:
                target_vector = next((v for v in vectors if v.get("id") == vector_id), None)
                
                if target_vector:
                    display_vector_detail(target_vector)
                else:
                    console.print(f"[yellow]Vector with ID '{vector_id}' not found.[/yellow]")

def display_general_results(results, verbose=False):
    """Display general information about results."""
    # Display all top-level keys
    console.print("[bold]Result Structure[/bold]")
    for key, value in results.items():
        if isinstance(value, dict):
            console.print(f"{key}: Dictionary with {len(value)} items")
        elif isinstance(value, list):
            console.print(f"{key}: List with {len(value)} items")
        else:
            console.print(f"{key}: {value}")
    
    # Display full results if verbose
    if verbose:
        console.print("\n[bold]Full Results (JSON):[/bold]")
        console.print_json(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 