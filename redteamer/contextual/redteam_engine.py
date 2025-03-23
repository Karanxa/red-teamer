"""
Contextual Red Team Engine for evaluating chatbot security.

This module provides a specialized engine for red teaming chatbots using
contextually-aware adversarial prompts.
"""

import os
import json
import logging
import time
import random
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from redteamer.contextual.prompt_generator import ContextualPromptGenerator
from redteamer.contextual.chatbot_connector import ChatbotConnector
from redteamer.utils.evaluator import RuleBasedEvaluator, ModelBasedEvaluator

class ContextualRedTeamEngine:
    """
    Engine for contextual red teaming of chatbots.
    
    This class handles the contextual red teaming process, including:
    1. Using the context of a chatbot to generate relevant adversarial prompts
    2. Sending prompts to the target chatbot
    3. Evaluating responses for security vulnerabilities
    4. Providing detailed results and reports
    """
    
    def __init__(
        self, 
        chatbot_curl: str, 
        context_file: str, 
        redteam_model: str = "karanxa/dravik",
        evaluator_model: Optional[str] = None,
        output_path: str = "results",
        verbose: bool = False
    ):
        """
        Initialize the contextual red team engine.
        
        Args:
            chatbot_curl: Curl command template for the target chatbot
            context_file: Path to file containing chatbot context description
            redteam_model: Model ID to use for generating adversarial prompts
            evaluator_model: Model ID to use for evaluating responses (optional)
            output_path: Path to save results
            verbose: Whether to print verbose output during scanning
        """
        self.logger = logging.getLogger(__name__)
        self.chatbot_curl = chatbot_curl
        self.context_file = context_file
        self.redteam_model = redteam_model
        self.evaluator_model = evaluator_model
        self.output_path = output_path
        self.verbose = verbose
        
        if self.verbose:
            self.logger.info(f"Initializing ContextualRedTeamEngine")
            self.logger.info(f"Red Team Model: {self.redteam_model}")
            self.logger.info(f"Evaluator Model: {self.evaluator_model}")
            self.logger.info(f"Context File: {self.context_file}")
            self.logger.info(f"Output Path: {self.output_path}")
        
        # Load chatbot context
        self.context = self._load_context()
        
        # Initialize components
        if self.verbose:
            self.logger.info(f"Initializing ContextualPromptGenerator with model: {redteam_model}")
        
        self.prompt_generator = ContextualPromptGenerator(
            model_id=redteam_model, 
            verbose=verbose
        )
        
        self.chatbot_connector = ChatbotConnector(
            curl_template=chatbot_curl,
            verbose=verbose
        )
        
        # Initialize evaluator
        self.rule_evaluator = RuleBasedEvaluator()
        self.model_evaluator = None
        if evaluator_model:
            from redteamer.utils.model_connector import ModelConnector
            model_connector = ModelConnector()
            self.model_evaluator = ModelBasedEvaluator(
                model_connector=model_connector,
                evaluator_model={
                    "provider": "openai" if "gpt" in evaluator_model else "anthropic",
                    "model_id": evaluator_model
                }
            )
        
        # Results storage
        self.results = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "chatbot_info": {
                "context_summary": self._get_context_summary(),
            },
            "model_info": {
                "redteam_model": redteam_model,
                "evaluator_model": evaluator_model
            },
            "prompts": [],
            "responses": [],
            "statistics": {}
        }
    
    def _load_context(self) -> str:
        """
        Load the chatbot context from a file.
        
        Returns:
            Chatbot context as a string
        """
        try:
            with open(self.context_file, 'r') as f:
                # Check if it's a JSON file
                if self.context_file.endswith('.json'):
                    data = json.load(f)
                    # Extract context from JSON fields if available
                    if isinstance(data, dict):
                        if 'context' in data:
                            return data['context']
                        elif 'description' in data:
                            return data['description']
                        else:
                            # Convert the entire JSON to a readable string format
                            return json.dumps(data, indent=2)
                    else:
                        return json.dumps(data)
                else:
                    # Treat as raw text
                    return f.read()
        except Exception as e:
            self.logger.error(f"Error loading context file: {e}")
            raise
    
    def _get_context_summary(self) -> str:
        """
        Get a short summary of the chatbot context.
        
        Returns:
            Summary of the context (first 500 characters)
        """
        # Get first 500 characters or first few lines
        summary = self.context[:500] + ("..." if len(self.context) > 500 else "")
        # Replace newlines with spaces for cleaner output
        summary = summary.replace("\n", " ").replace("  ", " ")
        return summary
    
    def generate_prompts(self, num_prompts: int = 20, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate contextual adversarial prompts.
        
        Args:
            num_prompts: Number of prompts to generate
            categories: Optional list of attack categories to focus on
            
        Returns:
            List of generated prompts
        """
        if self.verbose:
            self.logger.info(f"Generating {num_prompts} contextual prompts for chatbot red teaming")
        
        prompts = self.prompt_generator.generate_prompts(
            context=self.context,
            num_prompts=num_prompts,
            categories=categories
        )
        
        if self.verbose:
            self.logger.info(f"Generated {len(prompts)} contextual prompts")
            
            for i, prompt in enumerate(prompts):
                self.logger.debug(f"Prompt {i+1}: {prompt['prompt'][:100]}... [{prompt['category']}]")
        
        return prompts
    
    def test_chatbot(self, prompts: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Test the chatbot with the generated prompts.
        
        Args:
            prompts: List of generated prompts to test
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of test results
        """
        results = []
        
        # Test connection to the chatbot first
        if self.verbose:
            self.logger.info("Testing connection to the chatbot")
        
        connection_test = self.chatbot_connector.test_connection()
        if not connection_test["success"]:
            self.logger.error(f"Failed to connect to chatbot: {connection_test.get('error', 'Unknown error')}")
            raise ConnectionError(f"Failed to connect to chatbot: {connection_test.get('error', 'Unknown error')}")
        
        if self.verbose:
            self.logger.info(f"Connection to chatbot successful, response latency: {connection_test.get('latency', 'unknown')}s")
        
        # Test with generated prompts
        total_prompts = len(prompts)
        if self.verbose:
            self.logger.info(f"Testing chatbot with {total_prompts} prompts using {max_workers} workers")
        
        # Save the prompts to results
        self.results["prompts"] = prompts
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(self._test_single_prompt, prompt): prompt 
                for prompt in prompts
            }
            
            completed = 0
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed += 1
                    if self.verbose:
                        successful = result.get("evaluation", {}).get("success", False)
                        status = "[Success]" if successful else "[Failed]"
                        self.logger.info(f"Progress: {completed}/{total_prompts} - {status} {prompt['category']}: {prompt['prompt'][:50]}...")
                        
                except Exception as e:
                    self.logger.error(f"Error testing prompt: {e}")
                    results.append({
                        "prompt": prompt,
                        "error": str(e),
                        "success": False
                    })
        
        # Record total elapsed time
        elapsed_time = time.time() - start_time
        if self.verbose:
            self.logger.info(f"Completed testing {total_prompts} prompts in {elapsed_time:.2f} seconds")
            
        # Save detailed results
        self.results["responses"] = results
        self.results["statistics"] = self._calculate_statistics(results)
        
        return results
    
    def _test_single_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single prompt against the chatbot.
        
        Args:
            prompt: Prompt to test
            
        Returns:
            Test result
        """
        prompt_text = prompt["prompt"]
        category = prompt["category"]
        success_criteria = prompt["success_criteria"]
        
        try:
            # Send the prompt to the chatbot
            response = self.chatbot_connector.send_prompt(prompt_text)
            
            # Evaluate the response
            if self.model_evaluator:
                # Use model-based evaluation if available
                evaluation_result = self.model_evaluator.evaluate(
                    prompt=prompt_text,
                    response=response["response_text"],
                    attack_vector={
                        "prompt": prompt_text,
                        "category": category,
                        "success_criteria": success_criteria
                    }
                )
            else:
                # Use rule-based evaluation as fallback
                evaluation_result = self.rule_evaluator.evaluate(
                    prompt=prompt_text,
                    response=response["response_text"],
                    attack_vector={
                        "prompt": prompt_text,
                        "category": category,
                        "success_criteria": success_criteria
                    }
                )
            
            # Compile the result
            result = {
                "prompt_id": prompt.get("id", "unknown"),
                "prompt": prompt,
                "response": {
                    "text": response["response_text"],
                    "latency": response["latency"],
                    "tokens": response.get("token_count", {})
                },
                "evaluation": evaluation_result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error testing prompt: {e}")
            return {
                "prompt_id": prompt.get("id", "unknown"),
                "prompt": prompt,
                "error": str(e),
                "evaluation": {
                    "success": False,
                    "score": 0.0,
                    "reason": f"Error: {str(e)}"
                }
            }
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from the test results.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary of statistics
        """
        total_prompts = len(results)
        successful_attacks = sum(1 for r in results if r.get("evaluation", {}).get("success", False))
        success_rate = successful_attacks / total_prompts if total_prompts > 0 else 0
        
        # Group by category
        categories = {}
        for result in results:
            category = result.get("prompt", {}).get("category", "unknown")
            if category not in categories:
                categories[category] = {
                    "total": 0,
                    "successful": 0,
                    "success_rate": 0
                }
            
            categories[category]["total"] += 1
            if result.get("evaluation", {}).get("success", False):
                categories[category]["successful"] += 1
        
        # Calculate success rate per category
        for category in categories:
            cat_data = categories[category]
            cat_data["success_rate"] = cat_data["successful"] / cat_data["total"] if cat_data["total"] > 0 else 0
        
        # Calculate latency statistics
        latencies = [r.get("response", {}).get("latency", 0) for r in results if "error" not in r]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "total_prompts": total_prompts,
            "successful_attacks": successful_attacks,
            "success_rate": success_rate,
            "categories": categories,
            "average_latency": avg_latency
        }
    
    def run(self, num_prompts: int = 20, categories: Optional[List[str]] = None, max_workers: int = 4) -> Dict[str, Any]:
        """
        Run the complete contextual red teaming process.
        
        Args:
            num_prompts: Number of prompts to generate
            categories: Optional list of attack categories to focus on
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with the complete results
        """
        try:
            if self.verbose:
                self.logger.info("Starting contextual red teaming process")
                self.logger.info(f"Context summary: {self._get_context_summary()}")
            
            # Generate prompts
            prompts = self.generate_prompts(num_prompts, categories)
            
            # Test chatbot
            self.test_chatbot(prompts, max_workers)
            
            # Save results
            self._save_results()
            
            if self.verbose:
                self._print_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in contextual red teaming process: {e}")
            raise
    
    def _save_results(self) -> None:
        """Save the results to a file."""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contextual_redteam_{timestamp}.json"
            
            # Save results
            with open(output_path / filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            if self.verbose:
                self.logger.info(f"Results saved to {output_path / filename}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _print_summary(self) -> None:
        """Print a summary of the results."""
        stats = self.results.get("statistics", {})
        success_rate = stats.get("success_rate", 0) * 100
        
        self.logger.info("=" * 60)
        self.logger.info("CONTEXTUAL RED TEAM EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Chatbot Context: {self._get_context_summary()}")
        self.logger.info(f"Total Prompts: {stats.get('total_prompts', 0)}")
        self.logger.info(f"Successful Attacks: {stats.get('successful_attacks', 0)}")
        self.logger.info(f"Success Rate: {success_rate:.2f}%")
        self.logger.info(f"Average Latency: {stats.get('average_latency', 0):.2f} seconds")
        
        self.logger.info("-" * 60)
        self.logger.info("RESULTS BY CATEGORY")
        for category, data in stats.get("categories", {}).items():
            cat_success_rate = data.get("success_rate", 0) * 100
            self.logger.info(f"{category}: {data.get('successful', 0)}/{data.get('total', 0)} successful ({cat_success_rate:.2f}%)")
        
        self.logger.info("=" * 60) 