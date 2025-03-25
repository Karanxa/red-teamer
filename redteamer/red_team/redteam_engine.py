"""
Red Team Engine for evaluating LLM models.

This module provides the main engine for running red team evaluations.
"""

import os
import json
import logging
import tempfile
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from redteamer.utils.model_connector import ModelConnector, CustomModelConnector, OllamaConnector
from redteamer.utils.evaluator import RuleBasedEvaluator, ModelBasedEvaluator, HybridEvaluator

# Configure logging
logger = logging.getLogger(__name__)

class RedTeamEngine:
    """Red Team evaluation engine."""
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize the RedTeamEngine with a configuration file.
        
        Args:
            config_path: Path to configuration file.
            verbose: Whether to print verbose output.
        """
        self.config_path = config_path
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Set up evaluators
        self._setup_evaluators()
        
        # Initialize current vector and total vectors for progress tracking
        self.current_vector = 0
        self.total_vectors = 0
        
        # Initialize placeholder for current vector info
        self.current_vector_info = None
        
        # Initialize placeholder for last evaluation result
        self.last_evaluation_result = None
        
        # Initialize progress callback
        self.progress_callback = None
        
        # Initialize model connector
        self.model_connector = ModelConnector()
        
        # Initialize results
        self.results = {
            'vulnerabilities': [],
            'total_tests': 0,
            'metadata': self.config.get('metadata', {})
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load the red team configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary containing the red team configuration.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate config
            required_keys = ['name', 'models', 'dataset', 'parameters', 'evaluation', 'output']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
            
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_evaluators(self):
        """Set up evaluators based on configuration."""
        eval_config = self.config.get('evaluation', {})
        eval_type = eval_config.get('type', 'rule-based')
        
        # Create rule-based evaluator
        self.rule_evaluator = RuleBasedEvaluator()
        
        # If custom rules are defined, add them
        custom_rules = eval_config.get('rules', [])
        for rule in custom_rules:
            self.rule_evaluator.add_rule(rule)
        
        # Setup evaluator based on type
        if eval_type == 'rule-based':
            self.evaluator = self.rule_evaluator
            self.logger.info("Using rule-based evaluator")
        elif eval_type == 'model-based':
            # Check if evaluator model is defined
            evaluator_model = eval_config.get('evaluator_model')
            if not evaluator_model:
                # Use the first model in the config as the evaluator
                evaluator_model = self.config['models'][0]
                self.logger.warning("No evaluator model specified, using first model in config")
            
            self.model_evaluator = ModelBasedEvaluator(
                model_connector=self.model_connector,
                evaluator_model=evaluator_model
            )
            self.evaluator = self.model_evaluator
            self.logger.info(f"Using model-based evaluator with {evaluator_model.get('name', 'unspecified')} model")
        elif eval_type == 'hybrid':
            # Check if evaluator model is defined
            evaluator_model = eval_config.get('evaluator_model')
            if not evaluator_model:
                # Use the first model in the config as the evaluator
                evaluator_model = self.config['models'][0]
                self.logger.warning("No evaluator model specified, using first model in config")
            
            self.model_evaluator = ModelBasedEvaluator(
                model_connector=self.model_connector,
                evaluator_model=evaluator_model
            )
            
            # Get weights
            rule_weight = eval_config.get('rule_weight', 0.4)
            model_weight = eval_config.get('model_weight', 0.6)
            
            self.evaluator = HybridEvaluator(
                rule_evaluator=self.rule_evaluator,
                model_evaluator=self.model_evaluator,
                rule_weight=rule_weight,
                model_weight=model_weight
            )
            self.logger.info(f"Using hybrid evaluator with weights: rule={rule_weight}, model={model_weight}")
        else:
            self.logger.warning(f"Unknown evaluator type: {eval_type}, using rule-based as default")
            self.evaluator = self.rule_evaluator
    
    def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load a dataset from a file.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dataset dictionary
        """
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Check if dataset is in the correct format (with attack_vectors)
            if 'attack_vectors' not in dataset and 'prompts' in dataset:
                # Convert from prompts format
                attack_vectors = []
                for i, prompt in enumerate(dataset['prompts']):
                    attack_vectors.append({
                        'name': f"Vector {i+1}",
                        'prompt': prompt,
                        'category': 'generated',
                        'severity': 'medium',
                        'target_capability': 'safety',
                        'success_criteria': 'Model provides harmful, misleading, or unsafe content'
                    })
                    
                dataset['attack_vectors'] = attack_vectors
            
            return dataset
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def _sample_attack_vectors(self, dataset: Dict[str, Any], sample_size: int) -> List[Dict[str, Any]]:
        """
        Sample attack vectors from a dataset.
        
        Args:
            dataset: Dataset dictionary
            sample_size: Number of attack vectors to sample
            
        Returns:
            List of sampled attack vectors
        """
        attack_vectors = dataset.get('attack_vectors', [])
        
        if not attack_vectors:
            raise ValueError("No attack vectors found in dataset")
        
        # If sample size is larger than the number of attack vectors, use all
        if sample_size >= len(attack_vectors):
            return attack_vectors
        
        # Otherwise, sample randomly
        import random
        return random.sample(attack_vectors, sample_size)
    
    def _generate_response(self, vector: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response from a model to the vector prompt.
        
        Args:
            vector: The attack vector.
            model_config: Model configuration.
            
        Returns:
            Dictionary with the response and associated metadata.
        """
        try:
            # Extract prompt from vector
            prompt = vector.get("prompt", "")
            
            # Get evaluation parameters
            parameters = model_config.get("parameters", {})
            
            # Log the prompt and model
            self.logger.debug(f"Generating response from {model_config['provider']}/{model_config['model_id']} for vector {vector['id']}")
            
            # Initialize model connector if not already done
            if not hasattr(self, "model_connector"):
                self.model_connector = ModelConnector()
            
            # Generate response
            result = self.model_connector.generate_completion(model_config, prompt)
            
            # Extract necessary fields and format the result
            response = {
                "model": f"{model_config['provider']}/{model_config['model_id']}",
                "provider": model_config["provider"],
                "model_id": model_config["model_id"],
                "prompt": prompt,
                "response": result.get("response_text", ""),
                "success": result.get("success", True),
                "latency_ms": result.get("latency_ms", 0)
            }
            
            # Include token counts if available
            if "token_count" in result:
                response["token_count"] = result["token_count"]
            
            # Include any additional metrics
            if "extra_metrics" in result:
                response["metrics"] = result["extra_metrics"]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "model": f"{model_config['provider']}/{model_config['model_id']}",
                "provider": model_config["provider"],
                "model_id": model_config["model_id"],
                "prompt": vector.get("prompt", ""),
                "response": f"Error generating response: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_response(self, prompt: str, response_data: Dict[str, Any], vector: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a model response to determine if an attack was successful.
        
        Args:
            prompt: Prompt sent to the model
            response_data: Model response data dictionary
            vector: Attack vector
            
        Returns:
            Dictionary with evaluation result
        """
        # Extract the text response from the response_data
        response_text = response_data.get("response", "")
        
        # Use evaluator to determine if the response indicates a successful attack
        return self.evaluator.evaluate(response_text, vector)
    
    def _evaluate_vector(self, vector: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single attack vector against a specific model.
        
        Args:
            vector: Attack vector to evaluate
            model_config: Model configuration
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get the prompt from the vector
            prompt = vector.get("prompt", "")
            
            # Log vector being tested
            if self.verbose:
                self.logger.info(f"Testing vector {vector.get('id', 'unknown')}: {prompt[:100]}...")
            
            # Create the appropriate connector
            provider = model_config.get("provider")
            if provider == "custom":
                connector = CustomModelConnector()
            elif provider == "ollama":
                connector = OllamaConnector()
            else:
                connector = self.model_connector
            
            # Generate response from model
            response = self._generate_response(vector, model_config)
            
            # Evaluate the response
            evaluation = self._evaluate_response(prompt, response, vector)
            
            # Add response to the evaluation
            evaluation["response"] = response
            evaluation["prompt"] = prompt
            evaluation["vector_id"] = vector.get("id", "unknown")
            evaluation["model_id"] = model_config.get("model_id", "unknown")
            evaluation["provider"] = provider
            
            return evaluation
        except Exception as e:
            # Handle any exceptions during evaluation
            error_msg = f"Error evaluating vector {vector.get('id', 'unknown')}: {str(e)}"
            self.logger.error(error_msg)
            
            # Return a error result
            return {
                "success": False,
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "response": "",
                "prompt": vector.get("prompt", ""),
                "vector_id": vector.get("id", "unknown"),
                "model_id": model_config.get("model_id", "unknown"),
                "provider": model_config.get("provider", "unknown"),
                "error": str(e)
            }
    
    def run_redteam(self) -> Dict:
        """
        Run the red team evaluation.
        
        Returns:
            Dictionary containing red team evaluation results.
        """
        # Load dataset
        all_vectors = self._load_dataset(self.config['dataset'])
        vectors = self._sample_attack_vectors(all_vectors, self.config['parameters']['sample_size'])
        
        # Set total vectors for progress tracking
        self.total_vectors = len(vectors) * len(self.config['models'])
        self.current_vector = 0
        
        # Get red team parameters
        parallelism = self.config['parameters'].get('parallelism', 1)
        
        # Prepare results
        start_time = time.time()
        total_vectors = len(vectors)
        total_models = len(self.config['models'])
        self.logger.info(f"Starting red team evaluation with {total_vectors} vectors against {total_models} models")
        
        if self.verbose:
            print(f"\n[SCAN] Starting red team evaluation")
            print(f"[SCAN] Testing {total_vectors} attack vectors against {total_models} models")
            print(f"[SCAN] Parallelism: {parallelism}")
            print(f"[SCAN] Dataset: {self.config['dataset']}")
            print(f"[SCAN] Evaluation method: {self.config['evaluation'].get('method', 'rule-based')}")
            print(f"[SCAN] Output directory: {self.config['output'].get('path', 'results')}")
        
        # Run red team evaluation for each model
        all_results = []
        for model_index, model_config in enumerate(self.config['models']):
            model_name = f"{model_config.get('provider', 'unknown')}/{model_config.get('model_id', 'unknown')}"
            self.logger.info(f"Evaluating model: {model_name}")
            
            if self.verbose:
                print(f"\n[SCAN] Model {model_index+1}/{total_models}: {model_name}")
                print(f"[SCAN] Parameters: temperature={model_config.get('parameters', {}).get('temperature', 'default')}, max_tokens={model_config.get('parameters', {}).get('max_tokens', 'default')}")
                print(f"[SCAN] Testing {total_vectors} vectors against {model_name}...")
            
            # Evaluate vectors
            model_results = []
            if parallelism > 1 and False:  # Temporarily disable parallelism for better progress tracking
                # Run evaluations in parallel
                if self.verbose:
                    print(f"[SCAN] Running parallel evaluations with {parallelism} workers")
                
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {executor.submit(self._evaluate_vector, vector, model_config): vector for vector in vectors}
                    completed = 0
                    for future in as_completed(futures):
                        vector = futures[future]
                        try:
                            result = future.result()
                            model_results.append(result)
                            completed += 1
                            self.current_vector += 1
                            
                            # Update progress
                            if self.progress_callback:
                                self.progress_callback(self.current_vector, self.total_vectors)
                                
                            self.logger.debug(f"Evaluated vector {vector.get('id', '')} for model {model_name}")
                            if self.verbose:
                                print(f"[SCAN] Progress: {completed}/{total_vectors} vectors processed ({(completed/total_vectors)*100:.1f}%)")
                        except Exception as e:
                            self.logger.error(f"Error in evaluation: {e}")
                            if self.verbose:
                                print(f"[SCAN] Error processing vector: {str(e)}")
            else:
                # Run evaluations sequentially
                for i, vector in enumerate(vectors):
                    if self.verbose:
                        print(f"\n[SCAN] Processing vector {i+1}/{total_vectors}")
                    
                    # Update vector ID if missing for better logging
                    if "id" not in vector:
                        vector["id"] = f"adv-{i+1}"
                        
                    # Log which vector we're testing
                    vector_id = vector.get("id", f"adv-{i+1}")
                    self.logger.info(f"Testing vector {vector_id}: {vector.get('prompt', '')[:50]}...")
                    
                    result = self._evaluate_vector(vector, model_config)
                    model_results.append(result)
                    self.current_vector += 1
                    
                    # Update progress
                    if self.progress_callback:
                        self.progress_callback(self.current_vector, self.total_vectors)
                        
                    self.logger.debug(f"Evaluated vector {i+1}/{total_vectors} for model {model_name}")
                    
                    if self.verbose:
                        print(f"[SCAN] Progress: {i+1}/{total_vectors} vectors processed ({((i+1)/total_vectors)*100:.1f}%)")
            
            if self.verbose:
                success_count = sum(1 for r in model_results if r.get('success'))
                error_count = sum(1 for r in model_results if 'error' in r)
                if model_results:
                    print(f"\n[SCAN] Model {model_name} evaluation complete")
                    print(f"[SCAN] Vectors tested: {len(model_results)}")
                    print(f"[SCAN] Successful attacks: {success_count} ({success_count/len(model_results):.1%})")
                    print(f"[SCAN] Errors: {error_count} ({error_count/len(model_results):.1%})")
            
            all_results.extend(model_results)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Store results
        self.results['vulnerabilities'] = [r for r in all_results if r.get('success') is False]
        self.results['total_tests'] = len(all_results)
        
        # Generate red team summary
        summary = self._generate_summary(all_results)
        
        # Save results if specified
        if self.config['output'].get('save_results', True):
            self._save_results(all_results, summary)
            if self.verbose:
                output_path = self.config['output'].get('path', 'results')
                print(f"\n[SCAN] Results saved to: {output_path}")
        
        if self.verbose:
            print(f"\n[SCAN] Red team evaluation complete")
            print(f"[SCAN] Total elapsed time: {elapsed_time:.2f} seconds")
            print(f"[SCAN] Total vectors evaluated: {summary['sampled_vectors']}")
            print(f"[SCAN] Overall attack success rate: {summary['overall']['success_rate']:.2%}")
            print(f"[SCAN] Models evaluated: {len(summary['models'])}")
        
        return {
            'results': all_results,
            'summary': summary,
            'elapsed_time': elapsed_time
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with summary information
        """
        total_count = len(results)
        vulnerability_count = 0
        model_stats = {}
        
        for result in results:
            # Check if result indicates vulnerability
            if result.get("success", False):
                vulnerability_count += 1
            
            # Extract model and provider - be defensive about missing keys
            model_id = result.get("model_id", "unknown")
            provider = result.get("provider", "unknown")
            model_key = f"{provider}/{model_id}"
            
            # Update model stats
            if model_key not in model_stats:
                model_stats[model_key] = {
                    "total": 0,
                    "vulnerable": 0,
                    "confidence_sum": 0.0,
                    "top_vulnerabilities": []
                }
            
            model_stats[model_key]["total"] += 1
            
            if result.get("success", False):
                model_stats[model_key]["vulnerable"] += 1
                model_stats[model_key]["confidence_sum"] += result.get("confidence", 0.0)
                
                # Add to top vulnerabilities if notable
                if result.get("confidence", 0.0) > 0.7:
                    # Create a simplified version for the summary
                    vuln_entry = {
                        "prompt": result.get("prompt", ""),
                        "response": result.get("response", ""),
                        "confidence": result.get("confidence", 0.0),
                        "vector_id": result.get("vector_id", "unknown")
                    }
                    model_stats[model_key]["top_vulnerabilities"].append(vuln_entry)
            
        # Calculate overall statistics
        vulnerability_rate = vulnerability_count / total_count if total_count > 0 else 0
        
        # Calculate statistics per model
        models_summary = []
        for model_key, stats in model_stats.items():
            vuln_rate = stats["vulnerable"] / stats["total"] if stats["total"] > 0 else 0
            avg_confidence = stats["confidence_sum"] / stats["vulnerable"] if stats["vulnerable"] > 0 else 0
            
            # Sort and limit top vulnerabilities
            top_vulns = sorted(
                stats["top_vulnerabilities"], 
                key=lambda x: x.get("confidence", 0.0), 
                reverse=True
            )[:5]  # Keep only top 5
            
            models_summary.append({
                "model": model_key,
                "total_tests": stats["total"],
                "vulnerable_count": stats["vulnerable"],
                "vulnerability_rate": vuln_rate,
                "avg_confidence": avg_confidence,
                "top_vulnerabilities": top_vulns
            })
        
        # Sort models by vulnerability rate
        models_summary = sorted(
            models_summary, 
            key=lambda x: x.get("vulnerability_rate", 0.0),
            reverse=True
        )
        
        # Create summary dictionary
        summary = {
            "total_tests": total_count,
            "vulnerability_count": vulnerability_count,
            "vulnerability_rate": vulnerability_rate,
            "models": models_summary
        }
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict) -> None:
        """
        Save red team evaluation results to a file.
        
        Args:
            results: List of red team evaluation results.
            summary: Red team evaluation summary.
        """
        output_config = self.config['output']
        output_dir = output_config.get('path', 'results')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on red team name and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        redteam_name = self.config['name'].replace(' ', '_').lower()
        filename = f"{redteam_name}_{timestamp}"
        
        # Anonymize responses if specified
        if output_config.get('anonymize', False):
            for result in results:
                if result.get('response'):
                    result['response'] = "*** REDACTED ***"
        
        # Save results
        results_data = {
            'redteam_config': self.config,
            'summary': summary,
            'results': results
        }
        
        results_path = os.path.join(output_dir, f"{filename}.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Red team evaluation results saved to {results_path}")
    
    def get_results(self) -> List[Dict]:
        """
        Get red team evaluation results.
        
        Returns:
            List of red team evaluation results.
        """
        return self.results['vulnerabilities']

    # For backwards compatibility
    def run_benchmark(self) -> Dict:
        """
        Backwards compatibility method for running the red team evaluation.
        
        Returns:
            Dictionary containing red team evaluation results.
        """
        return self.run_redteam() 