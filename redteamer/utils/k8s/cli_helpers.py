"""
CLI helper functions for Kubernetes integration.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from pathlib import Path

# Import conditionally to avoid hard dependency
try:
    from .job_manager import K8sJobManager, KUBERNETES_AVAILABLE
except ImportError:
    KUBERNETES_AVAILABLE = False

# Console for rich output
console = Console()
logger = logging.getLogger(__name__)

def check_kubernetes_available():
    """Check if Kubernetes client libraries are available."""
    if not KUBERNETES_AVAILABLE:
        console.print("[bold red]Kubernetes support is not available.[/bold red]")
        console.print("Install the required dependency with: [bold]pip install kubernetes[/bold]")
        return False
    return True

def get_k8s_config_from_env():
    """Get Kubernetes configuration from environment variables."""
    config = {
        "namespace": os.environ.get("REDTEAMER_K8S_NAMESPACE", "default"),
        "image": os.environ.get("REDTEAMER_K8S_IMAGE", "redteamer:latest"),
        "service_account": os.environ.get("REDTEAMER_K8S_SERVICE_ACCOUNT", None),
        "config_file": os.environ.get("KUBECONFIG", None),
        "context": os.environ.get("REDTEAMER_K8S_CONTEXT", None),
        "in_cluster": os.environ.get("REDTEAMER_K8S_IN_CLUSTER", "false").lower() in ["true", "1", "yes"],
        "job_ttl_seconds": int(os.environ.get("REDTEAMER_K8S_JOB_TTL_SECONDS", "3600"))
    }
    return config

def launch_k8s_redteam_job(
    config: Dict,
    job_name: Optional[str] = None,
    namespace: Optional[str] = None,
    image: Optional[str] = None,
    service_account: Optional[str] = None,
    parallelism: int = 4,
    wait: bool = False,
    wait_timeout: int = 3600,
    env_vars: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Launch a red team evaluation as a Kubernetes job.
    
    Args:
        config: Red team configuration
        job_name: Name for the job
        namespace: Kubernetes namespace
        image: Docker image for the job
        service_account: Kubernetes service account
        parallelism: Number of parallel pods
        wait: Whether to wait for job completion
        wait_timeout: Timeout when waiting for job completion
        env_vars: Environment variables to pass to the job
        verbose: Whether to log verbose information
        
    Returns:
        Dictionary with job information
    """
    if not check_kubernetes_available():
        return {"error": "Kubernetes support is not available"}
    
    # Get default config from environment
    k8s_config = get_k8s_config_from_env()
    
    # Override with provided values
    if namespace:
        k8s_config["namespace"] = namespace
    if image:
        k8s_config["image"] = image
    if service_account:
        k8s_config["service_account"] = service_account
    
    # Create job manager
    job_manager = K8sJobManager(
        namespace=k8s_config["namespace"],
        image=k8s_config["image"],
        job_ttl_seconds=k8s_config["job_ttl_seconds"],
        service_account=k8s_config["service_account"],
        config_file=k8s_config["config_file"],
        context=k8s_config["context"],
        in_cluster=k8s_config["in_cluster"],
        verbose=verbose
    )
    
    # Check for API keys in environment and add them to env vars if needed
    if not env_vars:
        env_vars = {}
    
    # Add API keys from environment if models require them
    api_key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    for model in config.get("models", []):
        if "api_key_env" in model and model["api_key_env"] not in env_vars:
            env_value = os.environ.get(model["api_key_env"])
            if env_value:
                env_vars[model["api_key_env"]] = env_value
    
    # Launch the job
    with console.status(f"Launching job in namespace {k8s_config['namespace']}..."):
        job_id = job_manager.launch_redteam_job(
            config=config,
            job_name=job_name,
            parallelism=parallelism,
            active_deadline_seconds=wait_timeout if wait else None,
            env_vars=env_vars
        )
    
    console.print(f"[green]Launched job [bold]{job_id}[/bold] in namespace {k8s_config['namespace']}[/green]")
    
    # Wait for job completion if requested
    if wait:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Waiting for job completion..."),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Running", total=None)
            status = job_manager.wait_for_job_completion(job_id, timeout_seconds=wait_timeout)
            
        if status["status"] == "succeeded":
            console.print(f"[green]Job [bold]{job_id}[/bold] completed successfully[/green]")
            
            # Get results
            results = job_manager.get_job_results(job_id)
            if results:
                console.print("[green]Results retrieved successfully[/green]")
                return {"job_id": job_id, "status": status, "results": results}
            else:
                console.print("[yellow]Could not retrieve results from job[/yellow]")
                return {"job_id": job_id, "status": status}
        else:
            console.print(f"[yellow]Job [bold]{job_id}[/bold] status: {status['status']}[/yellow]")
            return {"job_id": job_id, "status": status}
    
    return {"job_id": job_id}

def get_k8s_job_status(job_id: str, namespace: Optional[str] = None, verbose: bool = False) -> Dict:
    """
    Get the status of a Kubernetes red team job.
    
    Args:
        job_id: Job ID (name)
        namespace: Kubernetes namespace
        verbose: Whether to log verbose information
        
    Returns:
        Job status dictionary
    """
    if not check_kubernetes_available():
        return {"error": "Kubernetes support is not available"}
    
    # Get default config from environment
    k8s_config = get_k8s_config_from_env()
    if namespace:
        k8s_config["namespace"] = namespace
    
    # Create job manager
    job_manager = K8sJobManager(
        namespace=k8s_config["namespace"],
        verbose=verbose
    )
    
    # Get job status
    with console.status(f"Getting status for job {job_id}..."):
        status = job_manager.get_job_status(job_id)
    
    return status

def list_k8s_jobs(namespace: Optional[str] = None, verbose: bool = False) -> List[Dict]:
    """
    List Kubernetes red team jobs.
    
    Args:
        namespace: Kubernetes namespace
        verbose: Whether to log verbose information
        
    Returns:
        List of job status dictionaries
    """
    if not check_kubernetes_available():
        return [{"error": "Kubernetes support is not available"}]
    
    # Get default config from environment
    k8s_config = get_k8s_config_from_env()
    if namespace:
        k8s_config["namespace"] = namespace
    
    # Create job manager
    job_manager = K8sJobManager(
        namespace=k8s_config["namespace"],
        verbose=verbose
    )
    
    # List jobs
    with console.status(f"Listing jobs in namespace {k8s_config['namespace']}..."):
        jobs = job_manager.list_jobs()
    
    return jobs

def display_k8s_jobs(jobs: List[Dict]):
    """
    Display Kubernetes jobs in a formatted table.
    
    Args:
        jobs: List of job status dictionaries
    """
    if not jobs:
        console.print("[yellow]No jobs found[/yellow]")
        return
    
    table = Table(title="Kubernetes Red Team Jobs")
    table.add_column("Job Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created", style="blue")
    table.add_column("Pods (Active/Succeeded/Failed)", style="magenta")
    
    for job in jobs:
        pods = f"{job.get('active', 0)}/{job.get('succeeded', 0)}/{job.get('failed', 0)}"
        status_style = {
            "succeeded": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "blue",
            "not_found": "red"
        }.get(job.get("status", ""), "white")
        
        created_at = job.get("creation_time", "unknown")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        
        table.add_row(
            job.get("name", "unknown"),
            f"[{status_style}]{job.get('status', 'unknown')}[/{status_style}]",
            str(created_at),
            pods
        )
    
    console.print(table)

def delete_k8s_job(job_id: str, namespace: Optional[str] = None, verbose: bool = False) -> bool:
    """
    Delete a Kubernetes red team job.
    
    Args:
        job_id: Job ID (name)
        namespace: Kubernetes namespace
        verbose: Whether to log verbose information
        
    Returns:
        True if successful, False otherwise
    """
    if not check_kubernetes_available():
        console.print("[bold red]Kubernetes support is not available.[/bold red]")
        return False
    
    # Get default config from environment
    k8s_config = get_k8s_config_from_env()
    if namespace:
        k8s_config["namespace"] = namespace
    
    # Create job manager
    job_manager = K8sJobManager(
        namespace=k8s_config["namespace"],
        verbose=verbose
    )
    
    # Delete job
    with console.status(f"Deleting job {job_id}..."):
        success = job_manager.delete_job(job_id)
    
    if success:
        console.print(f"[green]Job [bold]{job_id}[/bold] deleted successfully[/green]")
    else:
        console.print(f"[yellow]Failed to delete job [bold]{job_id}[/bold][/yellow]")
    
    return success 