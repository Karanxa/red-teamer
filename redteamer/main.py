"""
Main entry point for the Red Teaming Framework.

This module provides the main programmatic interface for the framework, allowing
either direct imports for use in scripts or CLI-based usage.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from redteamer.red_team.redteam_engine import RedTeamEngine
from redteamer.dataset.dataset_manager import DatasetManager
from redteamer.reports.report_generator import ReportGenerator
from redteamer.utils.model_connector import ModelConnector
from redteamer.utils.evaluator import RuleBasedEvaluator, ModelBasedEvaluator, HybridEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("redteamer")

# Default model configurations
DEFAULT_MODELS = {
    "openai": {
        "gpt-4": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "OPENAI_API_KEY"
        },
        "gpt-3.5-turbo": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "OPENAI_API_KEY"
        }
    },
    "anthropic": {
        "claude-3-opus": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "claude-3-sonnet": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "claude-3-haiku": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "ANTHROPIC_API_KEY"
        }
    },
    "gemini": {
        "gemini-pro": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "api_key_env": "GOOGLE_API_KEY"
        },
        "gemini-1.5-pro": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "api_key_env": "GOOGLE_API_KEY"
        },
        "gemini-1.5-flash": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "api_key_env": "GOOGLE_API_KEY"
        }
    }
}

# Default benchmark parameters
DEFAULT_REDTEAM_PARAMS = {
    "sample_size": 50,
    "confidence_level": 0.95,
    "success_threshold": 0.8,
    "seed": 42,
    "parallelism": 4
}

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.setLevel(numeric_level)

def run_redteam(config_path: str, verbose: bool = False) -> Dict:
    """
    Run a red team evaluation using the specified configuration.
    
    Args:
        config_path: Path to the red team configuration file
        verbose: Whether to print verbose output during scanning
        
    Returns:
        Dictionary containing red team evaluation results
    """
    try:
        logger.info(f"Starting red team evaluation with configuration: {config_path}")
        engine = RedTeamEngine(config_path, verbose=verbose)
        results = engine.run_redteam()
        logger.info(f"Red team evaluation completed successfully")
        
        return results
    except Exception as e:
        logger.error(f"Error running red team evaluation: {e}")
        raise

def run_interactive_redteam(
    name: str = None,
    models_config: List[Dict] = None,
    dataset_path: str = None,
    sample_size: int = None,
    output_dir: str = "results",
    verbose: bool = False
) -> Dict:
    """
    Run a red team evaluation with interactive configuration.
    
    Args:
        name: Red team evaluation name
        models_config: List of model configurations
        dataset_path: Path to attack vector dataset
        sample_size: Number of vectors to sample
        output_dir: Directory to save results
        verbose: Whether to print verbose output during scanning
        
    Returns:
        Dictionary containing red team evaluation results
    """
    try:
        # Create default configuration
        config = {
            "name": name or "Interactive Red Team Evaluation",
            "models": models_config or [],
            "dataset": dataset_path or "examples/sample_attack_vectors.json",
            "parameters": DEFAULT_REDTEAM_PARAMS.copy(),
            "evaluation": {
                "method": "rule-based"
            },
            "output": {
                "format": "json",
                "include_responses": True,
                "anonymize": False
            }
        }
        
        if sample_size:
            config["parameters"]["sample_size"] = sample_size
        
        # Create temporary configuration file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
            json.dump(config, temp)
            temp_config_path = temp.name
        
        try:
            # Run red team evaluation
            logger.info(f"Starting interactive red team evaluation")
            engine = RedTeamEngine(temp_config_path, verbose=verbose)
            results = engine.run_redteam()
            logger.info(f"Red team evaluation completed successfully")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            return results
        finally:
            # Clean up temporary file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    except Exception as e:
        logger.error(f"Error running interactive red team evaluation: {e}")
        raise

def compare_redteam_evaluations(
    result1_path: str,
    result2_path: str,
    output_path: Optional[str] = None,
    output_format: str = "markdown"
) -> Dict:
    """
    Compare two red team evaluation results.
    
    Args:
        result1_path: Path to first red team evaluation results
        result2_path: Path to second red team evaluation results
        output_path: Path to save comparison report
        output_format: Report format (markdown, json, csv, pdf)
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        # Load results
        with open(result1_path, 'r') as f:
            result1 = json.load(f)
        
        with open(result2_path, 'r') as f:
            result2 = json.load(f)
        
        # Create report generator
        report_generator = ReportGenerator()
        
        # Generate comparison report
        comparison = report_generator.generate_comparison_report(
            result1, result2, output_path, output_format
        )
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing red team evaluations: {e}")
        raise

def create_dataset(name: str, description: str, output_path: str, 
                   author: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> Dict:
    """
    Create a new empty dataset.
    
    Args:
        name: Dataset name
        description: Dataset description
        output_path: Path to save the dataset
        author: Optional dataset author
        tags: Optional list of tags
        
    Returns:
        Dictionary containing the created dataset
    """
    try:
        logger.info(f"Creating new dataset: {name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        dataset_manager = DatasetManager()
        dataset = dataset_manager.create_dataset(
            name=name,
            description=description,
            author=author,
            tags=tags or []
        )
        
        # Save dataset
        dataset_manager.save_dataset(output_path)
        
        logger.info(f"Dataset created successfully: {output_path}")
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

def load_dataset(dataset_path: str) -> Dict:
    """
    Load a dataset from a file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary containing the loaded dataset
    """
    try:
        logger.info(f"Loading dataset: {dataset_path}")
        dataset_manager = DatasetManager(dataset_path)
        dataset = dataset_manager.get_dataset()
        
        logger.info(f"Dataset loaded successfully: {len(dataset.get('vectors', []))} vectors")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def add_vector(dataset_path: str, vector: Dict, output_path: Optional[str] = None) -> Dict:
    """
    Add a vector to a dataset.
    
    Args:
        dataset_path: Path to the dataset file
        vector: Vector to add
        output_path: Optional path to save the updated dataset
        
    Returns:
        Dictionary containing the updated dataset
    """
    try:
        logger.info(f"Adding vector to dataset: {dataset_path}")
        dataset_manager = DatasetManager(dataset_path)
        
        # Extract vector properties
        vector_id = dataset_manager.add_vector(
            prompt=vector.get("prompt", ""),
            category=vector.get("category", ""),
            severity=vector.get("severity", "medium"),
            target_capability=vector.get("target_capability", ""),
            success_criteria=vector.get("success_criteria", ""),
            tags=vector.get("tags", []),
            metadata=vector.get("metadata", {})
        )
        
        # Save dataset
        if output_path:
            save_path = output_path
        else:
            save_path = dataset_path
            
        dataset_manager.save_dataset(save_path)
        
        logger.info(f"Vector {vector_id} added successfully")
        return dataset_manager.get_dataset()
    except Exception as e:
        logger.error(f"Error adding vector: {e}")
        raise

def generate_report(results_path: str, output_path: str, 
                    report_format: str = "markdown",
                    template: Optional[str] = None) -> str:
    """
    Generate a report from benchmark results.
    
    Args:
        results_path: Path to the benchmark results file
        output_path: Path to save the report
        report_format: Format of the report (markdown, json, csv, pdf)
        template: Optional path to a custom report template
        
    Returns:
        Path to the generated report
    """
    try:
        logger.info(f"Generating {report_format} report from: {results_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Load benchmark results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Generate report
        report_generator = ReportGenerator()
        
        if template:
            with open(template, 'r') as f:
                template_content = f.read()
            report_path = report_generator.generate_report(
                results, output_path, report_format, template_content
            )
        else:
            report_path = report_generator.generate_report(
                results, output_path, report_format
            )
        
        logger.info(f"Report generated successfully: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

def test_model(
    provider: str, 
    model: str, 
    prompt: str, 
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key_env: Optional[str] = None
) -> Dict:
    """
    Test a single model with a prompt.
    
    Args:
        provider: Model provider (e.g., 'openai', 'anthropic')
        model: Model name
        prompt: Prompt to send to the model
        system_prompt: Optional system prompt
        temperature: Temperature setting (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        api_key_env: Environment variable containing the API key
        
    Returns:
        Dictionary containing the model's response
    """
    try:
        logger.info(f"Testing model: {provider}/{model}")
        
        # Set appropriate API key environment variable if not specified
        if not api_key_env:
            provider_lower = provider.lower()
            if provider_lower in DEFAULT_MODELS and model in DEFAULT_MODELS[provider_lower]:
                api_key_env = DEFAULT_MODELS[provider_lower][model]["api_key_env"]
            else:
                api_key_env = f"{provider_lower.upper()}_API_KEY"
        
        # Create model configuration
        model_config = {
            "model_id": model,
            "provider": provider,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "api_key_env": api_key_env
        }
        
        model_connector = ModelConnector()
        
        response = model_connector.generate_completion(
            model_config=model_config,
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        logger.info(f"Model test completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise

def evaluate_response(response_text: str, attack_vector: Dict, 
                     evaluator_type: str = "rule-based",
                     model_config: Optional[Dict] = None) -> Dict:
    """
    Evaluate a model's response to an attack vector.
    
    Args:
        response_text: Model's response text
        attack_vector: Attack vector information
        evaluator_type: Type of evaluator (rule-based, model-based, hybrid)
        model_config: Optional model configuration for model-based evaluation
        
    Returns:
        Dictionary containing the evaluation results
    """
    try:
        logger.info(f"Evaluating response using {evaluator_type} evaluator")
        
        # Create evaluator
        if evaluator_type == "rule-based":
            evaluator = RuleBasedEvaluator()
        elif evaluator_type in ["model-based", "hybrid"]:
            if not model_config:
                raise ValueError("Model configuration required for model-based or hybrid evaluation")
            
            model_connector = ModelConnector()
            model_evaluator = ModelBasedEvaluator(model_connector, model_config)
            
            if evaluator_type == "model-based":
                evaluator = model_evaluator
            else:
                rule_evaluator = RuleBasedEvaluator()
                evaluator = HybridEvaluator(rule_evaluator, model_evaluator)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")
        
        # Evaluate response
        evaluation = evaluator.evaluate(response_text, attack_vector)
        
        logger.info(f"Evaluation completed successfully")
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        raise 