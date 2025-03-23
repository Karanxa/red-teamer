"""
Red Teaming Framework for LLM security evaluation.

A comprehensive framework for evaluating LLM security through benchmarking,
dataset management, and detailed reporting.
"""

__version__ = "0.1.0"
__author__ = "Red Teaming Framework Team"

# Core components
from redteamer.red_team.redteam_engine import RedTeamEngine
from redteamer.dataset.dataset_manager import DatasetManager
from redteamer.reports.report_generator import ReportGenerator

# Evaluators
from redteamer.utils.evaluator import RuleBasedEvaluator, ModelBasedEvaluator, HybridEvaluator

# Main functions
from redteamer.main import (
    run_redteam,
    compare_redteam_evaluations,
    create_dataset,
    add_vector,
    generate_report,
    test_model,
    evaluate_response
)

__all__ = [
    "RedTeamEngine",
    "DatasetManager", 
    "ReportGenerator",
    "RuleBasedEvaluator",
    "ModelBasedEvaluator", 
    "HybridEvaluator",
    "run_redteam",
    "compare_redteam_evaluations",
    "create_dataset",
    "add_vector",
    "generate_report",
    "test_model",
    "evaluate_response"
]
