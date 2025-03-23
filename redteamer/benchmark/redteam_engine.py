"""
Red Team Engine for evaluating LLM security against attack vectors.
"""

import json
import logging
import time
import random
import os
import statistics
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
    
    def __init__(self, config_path: str):
        """
        Initialize the red team engine.
        
        Args:
            config_path: Path to the red team configuration file.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.model_connector = ModelConnector()
        self.results = []
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
        Evaluate a single attack vector against a model.
        
        Args:
            vector: The attack vector to evaluate.
            model_config: The model configuration.
            
        Returns:
            Dictionary containing evaluation results.
        """
        try:
            # Get the prompt from the vector
            prompt = vector.get('prompt', '')
            system_prompt = None
            
            # Check if system prompt exists
            if 'system_prompt' in vector:
                system_prompt = vector['system_prompt']
            
            # Generate model completion
            response_data = self.model_connector.generate_completion(
                model_config=model_config,
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Evaluate the response
            response_text = response_data.get('response_text', '')
            evaluation = self.evaluator.evaluate(response_text, vector)
            
            # Create result object
            result = {
                'vector_id': vector.get('id', ''),
                'model': model_config.get('name', ''),
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'response': response_text if self.config['output'].get('include_responses', True) else None,
                'success': evaluation.get('success', False),  # True means attack succeeded (model failed)
                'confidence': evaluation.get('confidence', 0.0),
                'explanation': evaluation.get('explanation', ''),
                'matched_rules': evaluation.get('matched_rules', []),
                'response_data': {
                    'tokens': response_data.get('tokens', 0),
                    'latency': response_data.get('latency', 0)
                }
            }
            
            if 'rule_evaluation' in evaluation:
                result['rule_evaluation'] = evaluation['rule_evaluation']
            
            if 'model_evaluation' in evaluation:
                result['model_evaluation'] = evaluation['model_evaluation']
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error evaluating vector {vector.get('id', '')}: {e}")
            # Return error result
            return {
                'vector_id': vector.get('id', ''),
                'model': model_config.get('name', ''),
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'response': None,
                'error': str(e),
                'success': None,
                'confidence': 0.0
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
        
        # Run red team evaluation for each model
        all_results = []
        for model_config in self.config['models']:
            model_name = model_config.get('name', 'unnamed')
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Evaluate vectors
            model_results = []
            if parallelism > 1:
                # Run evaluations in parallel
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {executor.submit(self._evaluate_vector, vector, model_config): vector for vector in vectors}
                    for future in as_completed(futures):
                        vector = futures[future]
                        try:
                            result = future.result()
                            model_results.append(result)
                            self.logger.debug(f"Evaluated vector {vector.get('id', '')} for model {model_name}")
                        except Exception as e:
                            self.logger.error(f"Error in evaluation: {e}")
            else:
                # Run evaluations sequentially
                for i, vector in enumerate(vectors):
                    result = self._evaluate_vector(vector, model_config)
                    model_results.append(result)
                    self.logger.debug(f"Evaluated vector {i+1}/{total_vectors} for model {model_name}")
            
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