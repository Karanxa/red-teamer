"""
Red Team Engine for evaluating LLM security against attack vectors.
"""

import json
import logging
import time
import random
import os
import statistics
import uuid
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from redteamer.utils.model_connector import ModelConnector
from redteamer.utils.evaluator import RuleBasedEvaluator, ModelBasedEvaluator, HybridEvaluator

class RedTeamEngine:
    """
    Engine for red teaming LLM security against attack vectors.
    
    This class handles the red teaming process, including:
    1. Loading models and attack vectors
    2. Running red team evaluations with various parameters
    3. Collecting and analyzing results
    4. Generating statistics and reports
    """
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize the red team engine.
        
        Args:
            config_path: Path to the red team configuration file.
            verbose: Whether to print verbose output during scanning.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.model_connector = ModelConnector()
        self.results = []
        self.verbose = verbose
        self._setup_evaluators()
    
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
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load the attack vector dataset.
        
        Returns:
            List of attack vectors.
        """
        dataset_path = self.config['dataset']
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            if 'vectors' not in data:
                raise ValueError("Dataset missing 'vectors' key")
            
            vectors = data['vectors']
            return vectors
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def _sample_vectors(self, vectors: List[Dict]) -> List[Dict]:
        """
        Sample vectors from the dataset.
        
        Args:
            vectors: List of all vectors.
            
        Returns:
            List of sampled vectors.
        """
        params = self.config['parameters']
        sample_size = params.get('sample_size')
        
        if not sample_size or sample_size >= len(vectors):
            return vectors
        
        # Set random seed if specified
        seed = params.get('seed')
        if seed is not None:
            random.seed(seed)
        
        # Sample vectors
        sampled = random.sample(vectors, sample_size)
        return sampled
    
    def _evaluate_vector(self, vector: Dict, model_config: Dict) -> Dict:
        """
        Evaluate a single vector against a model.
        
        Args:
            vector: The attack vector to evaluate.
            model_config: The model configuration.
            
        Returns:
            Dictionary containing evaluation results.
        """
        vector_id = vector.get('id', str(uuid.uuid4()))
        prompt = vector.get('prompt', '')
        system_prompt = vector.get('system_prompt', '')
        category = vector.get('category', 'unknown')
        
        if self.verbose:
            print(f"[SCAN] Processing vector ID: {vector_id}, Category: {category}")
            print(f"[SCAN] Prompt: {prompt[:150]}..." if len(prompt) > 150 else f"[SCAN] Prompt: {prompt}")
        
        # Call the model
        result = {
            'vector_id': vector_id,
            'model': model_config.get('model_id', 'unknown'),
            'provider': model_config.get('provider', 'unknown'),
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            model_response = self.model_connector.generate_completion(
                model_config=model_config,
                prompt=prompt,
                system_prompt=system_prompt
            ).get('response_text', '')
            
            if self.verbose:
                print(f"[SCAN] Response received ({len(model_response)} chars)")
            
            result['response'] = model_response
            
            # Evaluate the response
            evaluation_result = self._evaluate_response(prompt, system_prompt, model_response, vector, model_config)
            result.update(evaluation_result)
            
            if self.verbose:
                success = result.get('success')
                status = "SUCCEEDED" if success else "FAILED"
                print(f"[SCAN] Attack {status} with confidence {result.get('confidence', 0):.2f}")
                if result.get('matched_rules'):
                    print(f"[SCAN] Matched rules: {', '.join(result.get('matched_rules', []))}")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error evaluating vector {vector_id}: {error_msg}")
            result['error'] = error_msg
            result['success'] = False
            
            if self.verbose:
                print(f"[SCAN] Error: {error_msg}")
        
        return result
    
    def _evaluate_response(self, prompt: str, system_prompt: str, model_response: str, vector: Dict, model_config: Dict) -> Dict:
        """
        Evaluate a model's response to determine if it represents a successful attack.
        
        Args:
            prompt: The prompt that was sent to the model.
            system_prompt: The system prompt that was sent to the model.
            model_response: The model's response.
            vector: The attack vector details.
            model_config: The model configuration.
            
        Returns:
            Dictionary containing evaluation results.
        """
        try:
            # Prepare attack vector with prompt included
            attack_vector = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "id": vector.get('id', ''),
                "category": vector.get('category', 'unknown'),
                "severity": vector.get('severity', 'medium'),
                "target_capability": vector.get('target_capability', 'unknown'),
                "success_criteria": vector.get('success_criteria', '')
            }
            
            # Get evaluation result from appropriate evaluator
            if hasattr(self, 'evaluator') and self.evaluator:
                evaluation_result = self.evaluator.evaluate(model_response, attack_vector)
            else:
                # Fallback to rule-based evaluation if no evaluator is set
                if hasattr(self, 'rule_evaluator'):
                    evaluation_result = self.rule_evaluator.evaluate(model_response, attack_vector)
                else:
                    # Default result if no evaluator is available
                    evaluation_result = {
                        "success": False,
                        "confidence": 0.0,
                        "matched_rules": [],
                        "explanation": "No evaluator available"
                    }
                
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def run_redteam(self) -> Dict:
        """
        Run the red team evaluation.
        
        Returns:
            Dictionary containing red team evaluation results.
        """
        # Load dataset
        all_vectors = self._load_dataset()
        vectors = self._sample_vectors(all_vectors)
        
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
            if parallelism > 1:
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
                    
                    result = self._evaluate_vector(vector, model_config)
                    model_results.append(result)
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
        self.results = all_results
        
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
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """
        Generate a summary of red team evaluation results.
        
        Args:
            results: List of red team evaluation results.
            
        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            'redteam_name': self.config['name'],
            'timestamp': datetime.now().isoformat(),
            'total_vectors': len(self._load_dataset()),
            'sampled_vectors': len({r['vector_id'] for r in results}),
            'models': {},
            'overall': {
                'success_rate': 0,
                'failure_rate': 0,
                'error_rate': 0
            }
        }
        
        # Group results by model
        model_results = {}
        for result in results:
            model = result['model']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        # Calculate statistics for each model
        total_success = 0
        for model, model_data in model_results.items():
            valid_results = [r for r in model_data if r['success'] is not None]
            success_count = sum(1 for r in valid_results if r['success'])
            error_count = sum(1 for r in model_data if 'error' in r)
            
            if valid_results:
                success_rate = success_count / len(valid_results)
                failure_rate = 1 - success_rate
            else:
                success_rate = 0
                failure_rate = 0
                
            if model_data:
                error_rate = error_count / len(model_data)
            else:
                error_rate = 0
            
            confidence_scores = [r['confidence'] for r in valid_results if 'confidence' in r]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            # Calculate response metrics
            response_tokens = [r['response_data'].get('tokens', 0) for r in model_data if 'response_data' in r]
            avg_tokens = statistics.mean(response_tokens) if response_tokens else 0
            
            response_latency = [r['response_data'].get('latency', 0) for r in model_data if 'response_data' in r]
            avg_latency = statistics.mean(response_latency) if response_latency else 0
            
            # Store model statistics
            summary['models'][model] = {
                'vectors_evaluated': len(model_data),
                'success_count': success_count,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'error_rate': error_rate,
                'avg_confidence': avg_confidence,
                'avg_tokens': avg_tokens,
                'avg_latency': avg_latency
            }
            
            total_success += success_count
        
        # Calculate overall statistics
        valid_results = [r for r in results if r['success'] is not None]
        if valid_results:
            overall_success_rate = total_success / len(valid_results)
            overall_failure_rate = 1 - overall_success_rate
        else:
            overall_success_rate = 0
            overall_failure_rate = 0
            
        error_count = sum(1 for r in results if 'error' in r)
        if results:
            overall_error_rate = error_count / len(results)
        else:
            overall_error_rate = 0
        
        summary['overall']['success_rate'] = overall_success_rate
        summary['overall']['failure_rate'] = overall_failure_rate
        summary['overall']['error_rate'] = overall_error_rate
        
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
        return self.results

    # For backwards compatibility
    def run_benchmark(self) -> Dict:
        """
        Backwards compatibility method for running the red team evaluation.
        
        Returns:
            Dictionary containing red team evaluation results.
        """
        return self.run_redteam() 