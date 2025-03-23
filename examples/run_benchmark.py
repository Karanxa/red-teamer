#!/usr/bin/env python
"""
Example script demonstrating how to use the Red Teaming Framework to run a benchmark.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to path to import redteamer module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redteamer.main import run_benchmark, generate_report, setup_logging

def main():
    """Main function to run a benchmark and generate a report."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run a red teaming benchmark')
    parser.add_argument('--config', '-c', required=True, help='Path to benchmark configuration file')
    parser.add_argument('--output', '-o', default='results', help='Directory to save results and reports')
    parser.add_argument('--format', '-f', default='markdown', choices=['markdown', 'json', 'csv', 'pdf'],
                        help='Report format')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', help='Path to log file')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create result file path
    config_name = os.path.basename(args.config).split('.')[0]
    results_path = os.path.join(args.output, f"{config_name}_results.json")
    report_path = os.path.join(args.output, f"{config_name}_report.{args.format}")
    
    # Run benchmark
    print(f"Running benchmark with configuration: {args.config}")
    benchmark_results = run_benchmark(args.config)
    
    # Save results to file
    with open(results_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Benchmark results saved to: {results_path}")
    
    # Generate report
    print(f"Generating {args.format} report...")
    generate_report(results_path, report_path, args.format)
    
    print(f"Report generated: {report_path}")
    print("\nBenchmark Summary:")
    print(f"- Models evaluated: {len(benchmark_results.get('summary', {}).get('models', {}))}")
    print(f"- Vectors tested: {benchmark_results.get('summary', {}).get('sampled_vectors', 0)}")
    print(f"- Overall success rate: {benchmark_results.get('summary', {}).get('overall', {}).get('success_rate', 0):.2%}")
    print(f"- Elapsed time: {benchmark_results.get('elapsed_time', 0):.2f} seconds")
    
    # Print model-specific results
    print("\nModel Results:")
    for model_name, model_stats in benchmark_results.get('summary', {}).get('models', {}).items():
        print(f"- {model_name}:")
        print(f"  - Success rate: {model_stats.get('success_rate', 0):.2%}")
        print(f"  - Avg confidence: {model_stats.get('avg_confidence', 0):.2f}")
        print(f"  - Vectors evaluated: {model_stats.get('vectors_evaluated', 0)}")

if __name__ == "__main__":
    main() 