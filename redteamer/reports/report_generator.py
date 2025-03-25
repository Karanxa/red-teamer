"""
Report generator for LLM red teaming benchmarks.
"""

import os
import json
import csv
import datetime
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np

class ReportGenerator:
    """
    Generates reports from benchmark results.
    
    Features:
    - Multiple output formats (markdown, JSON, CSV, PDF)
    - Visualization of benchmark results
    - Comparative analysis between benchmark runs
    - Actionable remediation recommendations
    """
    
    def __init__(self, benchmark_results: Dict):
        """
        Initialize a ReportGenerator.
        
        Args:
            benchmark_results: Benchmark results to generate a report for.
        """
        self.logger = logging.getLogger(__name__)
        self.results = benchmark_results
        
        # Get package directory for templates
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
            self.logger.warning(f"Created template directory: {self.template_dir}")
            
            # Create default template
            with open(os.path.join(self.template_dir, "report.md.j2"), "w") as f:
                f.write(DEFAULT_MARKDOWN_TEMPLATE)
        
        # Set up jinja environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def generate_report(self, output_path: str, format: str = "markdown") -> str:
        """
        Generate a report from benchmark results.
        
        Args:
            output_path: Path to save the report.
            format: Format of the report (markdown, json, csv, pdf).
            
        Returns:
            Path to the generated report.
        """
        format = format.lower()
        
        if format == "markdown" or format == "md":
            return self._generate_markdown_report(output_path)
        elif format == "json":
            return self._generate_json_report(output_path)
        elif format == "csv":
            return self._generate_csv_report(output_path)
        elif format == "pdf":
            return self._generate_pdf_report(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, output_path: str) -> str:
        """
        Generate a markdown report.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Path to the generated report.
        """
        try:
            # Load template
            template = self.env.get_template("report.md.j2")
            
            # Prepare data
            data = self._prepare_report_data()
            
            # Add safety check for empty models
            if not data["models"]:
                self.logger.warning("No models found in the report data")
                # Add a placeholder model to avoid template errors
                data["models"] = [{"name": "unknown", "provider": "unknown", "model_id": "unknown"}]
            
            # Render template
            report_content = template.render(**data)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            self.logger.info(f"Generated markdown report: {output_path}")
            
            # Generate images if needed
            self._generate_visualizations(os.path.dirname(os.path.abspath(output_path)))
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error generating markdown report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_json_report(self, output_path: str) -> str:
        """
        Generate a JSON report.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Path to the generated report.
        """
        try:
            # Prepare data
            data = self._prepare_report_data()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Generated JSON report: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            raise
    
    def _generate_csv_report(self, output_path: str) -> str:
        """
        Generate a CSV report.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Path to the generated report.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Create a DataFrame from the results
            examples_df = pd.DataFrame([
                {
                    "test_id": example["test_id"],
                    "prompt": example["prompt"],
                    "category": example["evaluation"]["attack_category"],
                    "severity": example["evaluation"]["severity"],
                    "target_capability": example["evaluation"]["target_capability"],
                    **{f"{resp['model_id']}_success": resp["success"] for resp in example["responses"]},
                    **{f"{resp['model_id']}_latency_ms": resp.get("latency_ms", 0) for resp in example["responses"]}
                }
                for example in self.results["examples"]
            ])
            
            # Save to file
            examples_df.to_csv(output_path, index=False)
            
            # Create a summary file
            summary_path = os.path.splitext(output_path)[0] + "_summary.csv"
            summary_data = []
            
            # Add model success rates
            for model_id, rate in self.results["summary"]["model_success_rates"].items():
                confidence_interval = self.results["summary"]["statistical_analysis"]["confidence_intervals"].get(model_id, [0, 0])
                summary_data.append({
                    "type": "model_success_rate",
                    "name": model_id,
                    "value": rate,
                    "confidence_low": confidence_interval[0],
                    "confidence_high": confidence_interval[1]
                })
            
            # Add category success rates
            for category, rate in self.results["summary"]["category_success_rates"].items():
                summary_data.append({
                    "type": "category_success_rate",
                    "name": category,
                    "value": rate
                })
            
            # Add overall success rate
            summary_data.append({
                "type": "overall_success_rate",
                "name": "overall",
                "value": self.results["summary"]["overall_success_rate"]
            })
            
            # Save summary to file
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            
            self.logger.info(f"Generated CSV report: {output_path} and {summary_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
            raise
    
    def _generate_pdf_report(self, output_path: str) -> str:
        """
        Generate a PDF report.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Path to the generated report.
        """
        try:
            # Generate markdown first
            md_path = os.path.splitext(output_path)[0] + ".md"
            self._generate_markdown_report(md_path)
            
            # Try to convert to PDF if possible
            try:
                import weasyprint
                
                # Load markdown as HTML
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                
                try:
                    import markdown
                    html_content = markdown.markdown(
                        md_content,
                        extensions=['tables', 'fenced_code']
                    )
                except ImportError:
                    self.logger.warning("Could not convert markdown to HTML properly. Install 'markdown' for better results.")
                    # Basic conversion
                    html_content = f"<pre>{md_content}</pre>"
                
                # Add basic styling
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Benchmark Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .success {{ color: green; }}
                        .failure {{ color: red; }}
                        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """
                
                # Generate PDF
                weasyprint.HTML(string=html_content).write_pdf(output_path)
                self.logger.info(f"Generated PDF report: {output_path}")
                return output_path
            except ImportError:
                self.logger.warning("Could not generate PDF report. Please install weasyprint: pip install weasyprint")
                return md_path
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            raise
    
    def _prepare_report_data(self) -> Dict:
        """
        Prepare data for the report.
        
        Returns:
            Dictionary with data for the report template.
        """
        # Handle different result formats - both the new redteam_engine format and older format
        if "metadata" in self.results:
            # Basic benchmark information from metadata
            benchmark_info = {
                "id": self.results["metadata"].get("benchmark_id", "unknown"),
                "name": self.results["metadata"].get("name", "Red Team Evaluation"),
                "description": self.results["metadata"].get("description", ""),
                "version": self.results["metadata"].get("version", "1.0"),
                "created_at": self.results["metadata"].get("created_at", datetime.datetime.now().isoformat()),
                "parameters": self.results["metadata"].get("parameters", {})
            }
        elif "redteam_config" in self.results:
            # Extract from redteam config
            config = self.results["redteam_config"]
            benchmark_info = {
                "id": f"redteam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": config.get("name", "Red Team Evaluation"),
                "description": "Security evaluation of language models",
                "version": "1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "parameters": config.get("parameters", {})
            }
        else:
            # Create default information
            benchmark_info = {
                "id": f"redteam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": "Red Team Evaluation",
                "description": "Security evaluation of language models",
                "version": "1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "parameters": {}
            }
        
        # Model information - extract from different possible formats
        if "models_tested" in self.results:
            models = self.results["models_tested"]
        elif "summary" in self.results and "models" in self.results["summary"]:
            # Extract from the new format summary
            models = []
            for model_info in self.results["summary"]["models"]:
                models.append({
                    "name": model_info.get("model", "unknown"),
                    "provider": model_info.get("model", "unknown").split("/")[0] if "/" in model_info.get("model", "unknown") else "unknown",
                    "model_id": model_info.get("model", "unknown").split("/")[1] if "/" in model_info.get("model", "unknown") else model_info.get("model", "unknown"),
                    "parameters": {},
                    "success_rate": model_info.get("vulnerability_rate", 0.0),
                    "total_tests": model_info.get("total_tests", 0)
                })
        elif "redteam_config" in self.results and "models" in self.results["redteam_config"]:
            # Extract from redteam config
            models = []
            for model_config in self.results["redteam_config"]["models"]:
                models.append({
                    "name": f"{model_config.get('provider', 'unknown')}/{model_config.get('model_id', 'unknown')}",
                    "provider": model_config.get("provider", "unknown"),
                    "model_id": model_config.get("model_id", "unknown"),
                    "parameters": model_config.get("parameters", {})
                })
        else:
            # Create default model info
            models = [{"name": "unknown", "provider": "unknown", "model_id": "unknown"}]
        
        # Summary statistics - extract from different possible formats
        if "summary" in self.results:
            if "models" in self.results["summary"]:
                # New format summary
                summary = {
                    "overall_success_rate": 1.0 - self.results["summary"].get("vulnerability_rate", 0.0),
                    "total_tests": self.results["summary"].get("total_tests", 0),
                    "model_success_rates": {
                        m["model"]: 1.0 - m.get("vulnerability_rate", 0.0) 
                        for m in self.results["summary"]["models"]
                    },
                    "category_success_rates": {},  # Not available in new format
                    "statistical_analysis": {
                        "confidence_level": benchmark_info["parameters"].get("confidence_level", 0.95),
                        "confidence_intervals": {}
                    }
                }
            else:
                # Old format summary
                summary = self.results["summary"]
        else:
            # Create default summary
            summary = {
                "overall_success_rate": 0.0,
                "total_tests": 0,
                "model_success_rates": {},
                "category_success_rates": {},
                "statistical_analysis": {
                    "confidence_level": 0.95,
                    "confidence_intervals": {}
                }
            }
        
        # Examples (test cases) - extract from different possible formats
        if "examples" in self.results:
            examples = self.results["examples"]
        elif "results" in self.results:
            # Convert results to examples format
            examples = []
            for i, result in enumerate(self.results["results"]):
                examples.append({
                    "test_id": result.get("vector_id", f"test_{i}"),
                    "prompt": result.get("prompt", ""),
                    "evaluation": {
                        "attack_category": "unknown",
                        "severity": "medium",
                        "target_capability": "unknown",
                        "success": result.get("success", False),
                        "confidence": result.get("confidence", 0.0),
                        "explanation": result.get("explanation", "")
                    },
                    "responses": [{
                        "model_id": result.get("model_id", "unknown"),
                        "response": result.get("response", ""),
                        "success": result.get("success", False),
                        "latency_ms": 0,
                        "tokens": 0
                    }]
                })
        else:
            examples = []
        
        # Group examples by category
        examples_by_category = {}
        for example in examples:
            category = example.get("evaluation", {}).get("attack_category", "unknown")
            if category not in examples_by_category:
                examples_by_category[category] = []
            examples_by_category[category].append(example)
        
        # Calculate additional statistics
        
        # Success rate by severity
        severity_success_rates = {}
        severity_counts = {}
        
        for example in examples:
            severity = example.get("evaluation", {}).get("severity", "medium")
            
            if severity not in severity_success_rates:
                severity_success_rates[severity] = 0
                severity_counts[severity] = 0
            
            for response in example.get("responses", []):
                success = response.get("success", False)
                severity_success_rates[severity] += 1 if success else 0
                severity_counts[severity] += 1
        
        for severity, count in severity_counts.items():
            if count > 0:
                severity_success_rates[severity] = severity_success_rates[severity] / count
        
        # Calculate most vulnerable categories from category success rates
        category_rates = summary.get("category_success_rates", {})
        categories = [(cat, rate) for cat, rate in category_rates.items()]
        categories.sort(key=lambda x: x[1])
        most_vulnerable_categories = categories[:3] if len(categories) >= 3 else categories
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations()
        
        return {
            "benchmark": benchmark_info,
            "models": models,
            "summary": summary,
            "examples": examples,
            "examples_by_category": examples_by_category,
            "severity_success_rates": severity_success_rates,
            "most_vulnerable_categories": most_vulnerable_categories,
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """
        Generate recommendations based on benchmark results.
        
        Returns:
            List of recommendation dictionaries.
        """
        recommendations = []
        
        # Check what format we're dealing with
        if "summary" not in self.results:
            # No summary available, return basic recommendations
            recommendations.append({
                "category": "Analysis",
                "title": "Perform detailed analysis",
                "description": "Conduct a detailed analysis of the test results to identify security vulnerabilities.",
                "priority": "medium"
            })
            return recommendations
        
        # New format has a different structure
        if "models" in self.results["summary"]:
            # The new format with vulnerability_rate
            
            # Check overall performance
            vulnerability_rate = self.results["summary"].get("vulnerability_rate", 0.0)
            threshold = 0.2  # 20% vulnerability as a reasonable threshold
            
            if vulnerability_rate > threshold:
                recommendations.append({
                    "category": "General Security",
                    "title": "General security improvements needed",
                    "description": f"Overall vulnerability rate ({vulnerability_rate:.2%}) is above the target threshold "
                                 f"({threshold:.2%}). A comprehensive review of security measures is recommended.",
                    "priority": "high" if vulnerability_rate > 0.4 else "medium"
                })
            else:
                recommendations.append({
                    "category": "Monitoring",
                    "title": "Establish continuous monitoring",
                    "description": "Although current performance meets threshold requirements, establish continuous "
                                 "monitoring to detect potential regressions in future model versions.",
                    "priority": "low"
                })
            
            # Look at model performance differences
            models_info = self.results["summary"]["models"]
            if len(models_info) > 1:
                # Sort models by vulnerability rate
                models_info.sort(key=lambda x: x.get("vulnerability_rate", 0))
                
                best_model = models_info[0]
                worst_model = models_info[-1]
                
                best_rate = best_model.get("vulnerability_rate", 0)
                worst_rate = worst_model.get("vulnerability_rate", 0)
                
                if worst_rate - best_rate > 0.2:  # At least 20% difference
                    recommendations.append({
                        "category": "Model Selection",
                        "title": "Consider model replacement or configuration tuning",
                        "description": f"Model {best_model.get('model', 'unknown')} significantly outperformed {worst_model.get('model', 'unknown')} "
                                     f"(vulnerability rates: {best_rate:.2%} vs {worst_rate:.2%}). Consider replacing or reconfiguring "
                                     f"the underperforming model.",
                        "priority": "medium"
                    })
                
            # Add specific recommendations based on vulnerability types if available
            for model_info in models_info:
                if "top_vulnerabilities" in model_info and model_info["top_vulnerabilities"]:
                    model_name = model_info.get("model", "unknown")
                    recommendations.append({
                        "category": "Specific Vulnerabilities",
                        "title": f"Address specific vulnerabilities in {model_name}",
                        "description": f"Model {model_name} has specific vulnerability patterns that should be addressed. "
                                     f"Review the top vulnerabilities section for detailed examples.",
                        "priority": "high" if model_info.get("vulnerability_rate", 0) > 0.3 else "medium"
                    })
                    break  # Just add one general recommendation about specific vulnerabilities
            
            return recommendations
        
        # Old format handling follows
        
        # Look at the most vulnerable categories
        categories = [(cat, rate) for cat, rate in self.results["summary"].get("category_success_rates", {}).items()]
        categories.sort(key=lambda x: x[1])
        
        # Try to get the threshold from parameters
        threshold = 0.8  # Default threshold
        if "metadata" in self.results and "parameters" in self.results["metadata"]:
            threshold = self.results["metadata"]["parameters"].get("success_threshold", 0.8)
        elif "redteam_config" in self.results and "parameters" in self.results["redteam_config"]:
            threshold = self.results["redteam_config"]["parameters"].get("success_threshold", 0.8)
        
        # Add recommendations for the lowest performing categories
        for category, rate in categories[:3]:
            if rate < threshold:
                recommendations.append({
                    "category": "Vulnerability",
                    "title": f"Improve {category} defenses",
                    "description": f"Models demonstrated significant vulnerability to {category} attacks (success rate: {rate:.2%}). "
                                  f"Review and strengthen defenses against this attack vector.",
                    "priority": "high" if rate < 0.3 else "medium"
                })
        
        # Look at model performance differences
        model_rates = self.results["summary"].get("model_success_rates", {})
        if len(model_rates) > 1:
            model_items = list(model_rates.items())
            model_items.sort(key=lambda x: x[1])
            
            best_model, best_rate = model_items[-1]
            worst_model, worst_rate = model_items[0]
            
            if best_rate - worst_rate > 0.2:  # At least 20% difference
                recommendations.append({
                    "category": "Model Selection",
                    "title": "Consider model replacement or configuration tuning",
                    "description": f"Model {best_model} significantly outperformed {worst_model} "
                                  f"({best_rate:.2%} vs {worst_rate:.2%}). Consider replacing or reconfiguring "
                                  f"the underperforming model.",
                    "priority": "medium"
                })
        
        # Check overall performance
        overall_rate = self.results["summary"].get("overall_success_rate", 0.0)
        
        if overall_rate < threshold:
            recommendations.append({
                "category": "General Security",
                "title": "General security improvements needed",
                "description": f"Overall success rate ({overall_rate:.2%}) is below the target threshold "
                              f"({threshold:.2%}). A comprehensive review of security measures is recommended.",
                "priority": "high" if overall_rate < threshold * 0.7 else "medium"
            })
        else:
            recommendations.append({
                "category": "Monitoring",
                "title": "Establish continuous monitoring",
                "description": "Although current performance meets threshold requirements, establish continuous "
                              "monitoring to detect potential regressions in future model versions.",
                "priority": "low"
            })
        
        return recommendations
    
    def _generate_visualizations(self, output_dir: str) -> List[str]:
        """
        Generate visualizations for the report.
        
        Args:
            output_dir: Directory to save the visualizations.
            
        Returns:
            List of paths to generated visualization files.
        """
        try:
            # Create visualizations directory
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            generated_files = []
            
            # Set seaborn style
            sns.set_theme(style="whitegrid")
            
            # Check if we have the necessary data for visualizations
            if "summary" not in self.results:
                self.logger.warning("No summary data available for visualizations")
                return generated_files
            
            # Check which format we're dealing with
            if "models" in self.results["summary"]:
                # New format with vulnerability_rate
                
                # 1. Model vulnerability rates
                plt.figure(figsize=(10, 6))
                models_info = self.results["summary"]["models"]
                
                # Extract model names and rates
                models = [m.get("model", "unknown") for m in models_info]
                rates = [m.get("vulnerability_rate", 0.0) for m in models_info]
                
                # Sort by vulnerability rate (lowest first for better visualization)
                sorted_indices = sorted(range(len(rates)), key=lambda i: rates[i])
                models = [models[i] for i in sorted_indices]
                rates = [rates[i] for i in sorted_indices]
                
                # Plot
                ax = sns.barplot(x=rates, y=models, palette="viridis")
                ax.set_xlabel("Vulnerability Rate")
                ax.set_ylabel("Model")
                ax.set_title("Model Vulnerability Rates")
                
                # Add values to bars
                for i, rate in enumerate(rates):
                    ax.text(max(0.02, rate - 0.05), i, f"{rate:.2%}", va='center')
                
                # Save figure
                fig_path = os.path.join(vis_dir, "model_vulnerability_rates.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
                
                # 2. Test counts per model
                plt.figure(figsize=(10, 6))
                test_counts = [m.get("total_tests", 0) for m in models_info]
                
                # Use the same order from previous plot for consistency
                test_counts = [test_counts[i] for i in sorted_indices]
                
                # Plot
                ax = sns.barplot(x=test_counts, y=models, palette="rocket")
                ax.set_xlabel("Number of Tests")
                ax.set_ylabel("Model")
                ax.set_title("Tests per Model")
                
                # Add values to bars
                for i, count in enumerate(test_counts):
                    ax.text(max(1, count - 5), i, str(count), va='center')
                
                # Save figure
                fig_path = os.path.join(vis_dir, "tests_per_model.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
                
                # 3. Overall pie chart of vulnerability rate
                plt.figure(figsize=(8, 8))
                vulnerability_rate = self.results["summary"].get("vulnerability_rate", 0.0)
                success_rate = 1.0 - vulnerability_rate
                
                # Plot
                plt.pie(
                    [vulnerability_rate, success_rate],
                    labels=["Vulnerable", "Secure"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=["#FF9999", "#99FF99"]
                )
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title("Overall Security Assessment")
                
                # Save figure
                fig_path = os.path.join(vis_dir, "overall_security.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
                
                return generated_files
            
            # Old format handling follows
            
            # 1. Model success rates
            if "model_success_rates" in self.results["summary"]:
                plt.figure(figsize=(10, 6))
                model_rates = self.results["summary"]["model_success_rates"]
                models = list(model_rates.keys())
                rates = list(model_rates.values())
                
                # Sort by rate
                sorted_indices = sorted(range(len(rates)), key=lambda i: rates[i])
                models = [models[i] for i in sorted_indices]
                rates = [rates[i] for i in sorted_indices]
                
                # Plot
                ax = sns.barplot(x=rates, y=models, palette="viridis")
                ax.set_xlabel("Success Rate")
                ax.set_ylabel("Model")
                ax.set_title("Model Performance Against Attack Vectors")
                
                # Add values to bars
                for i, rate in enumerate(rates):
                    ax.text(max(0.02, rate - 0.05), i, f"{rate:.2%}", va='center')
                
                # Save figure
                fig_path = os.path.join(vis_dir, "model_success_rates.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
            
            # 2. Category success rates
            if "category_success_rates" in self.results["summary"]:
                plt.figure(figsize=(12, 6))
                category_rates = self.results["summary"]["category_success_rates"]
                categories = list(category_rates.keys())
                rates = list(category_rates.values())
                
                # Sort by rate
                sorted_indices = sorted(range(len(rates)), key=lambda i: rates[i])
                categories = [categories[i] for i in sorted_indices]
                rates = [rates[i] for i in sorted_indices]
                
                # Plot
                ax = sns.barplot(x=rates, y=categories, palette="rocket")
                ax.set_xlabel("Success Rate")
                ax.set_ylabel("Attack Category")
                ax.set_title("Success Rate by Attack Category")
                
                # Add values to bars
                for i, rate in enumerate(rates):
                    ax.text(max(0.02, rate - 0.05), i, f"{rate:.2%}", va='center')
                
                # Save figure
                fig_path = os.path.join(vis_dir, "category_success_rates.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
            
            # 3. Severity success rates
            if "examples" in self.results:
                # Calculate severity success rates
                severity_success_rates = {}
                severity_counts = {}
                
                for example in self.results["examples"]:
                    severity = example.get("evaluation", {}).get("severity", "medium")
                    
                    if severity not in severity_success_rates:
                        severity_success_rates[severity] = 0
                        severity_counts[severity] = 0
                    
                    for response in example.get("responses", []):
                        success = response.get("success", False)
                        severity_success_rates[severity] += 1 if success else 0
                        severity_counts[severity] += 1
                
                for severity, count in severity_counts.items():
                    if count > 0:
                        severity_success_rates[severity] = severity_success_rates[severity] / count
                
                if severity_success_rates:
                    plt.figure(figsize=(8, 6))
                    severities = list(severity_success_rates.keys())
                    rates = list(severity_success_rates.values())
                    
                    # Sort by severity (low, medium, high, critical)
                    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                    sorted_indices = sorted(range(len(severities)), key=lambda i: severity_order.get(severities[i].lower(), 99))
                    severities = [severities[i] for i in sorted_indices]
                    rates = [rates[i] for i in sorted_indices]
                    
                    # Plot
                    ax = sns.barplot(x=severities, y=rates, palette="YlOrRd")
                    ax.set_xlabel("Severity")
                    ax.set_ylabel("Success Rate")
                    ax.set_title("Success Rate by Attack Severity")
                    
                    # Add values to bars
                    for i, rate in enumerate(rates):
                        ax.text(i, rate + 0.02, f"{rate:.2%}", ha='center')
                    
                    # Save figure
                    fig_path = os.path.join(vis_dir, "severity_success_rates.png")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()
                    generated_files.append(fig_path)
            
            # 4. Overall success rate as a gauge chart
            if "overall_success_rate" in self.results["summary"]:
                plt.figure(figsize=(8, 5))
                overall_rate = self.results["summary"]["overall_success_rate"]
                
                # Create gauge chart
                gauge_angle = 180 * overall_rate  # 0 to 180 degrees
                
                # Plot
                ax = plt.subplot(1, 1, 1, polar=True)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_rlabel_position(0)
                
                # Set gauge limits
                ax.set_thetamin(0)
                ax.set_thetamax(180)
                
                # Add colored sections for different performance levels
                sections = [
                    (0, 60, "#FF9999", "Poor"),  # Red
                    (60, 120, "#FFCC99", "Moderate"),  # Orange
                    (120, 180, "#99FF99", "Good")  # Green
                ]
                
                for start_angle, end_angle, color, label in sections:
                    ax.bar(
                        np.deg2rad(np.linspace(start_angle, end_angle, 10)),
                        [1] * 10,
                        width=np.deg2rad(end_angle - start_angle) / 10,
                        color=color,
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.5
                    )
                
                # Add needle
                ax.plot([0, np.deg2rad(gauge_angle)], [0, 0.8], color='black', linewidth=2)
                ax.text(
                    np.deg2rad(gauge_angle), 0.85,
                    f"{overall_rate:.1%}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    fontweight='bold'
                )
                
                # Remove tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                
                # Add title
                plt.title("Overall Security Score", y=1.1, fontsize=16)
                
                # Add labels for sections
                for start_angle, end_angle, _, label in sections:
                    mid_angle = np.deg2rad((start_angle + end_angle) / 2)
                    ax.text(
                        mid_angle, 0.6,
                        label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=9
                    )
                
                # Save figure
                fig_path = os.path.join(vis_dir, "overall_score.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                generated_files.append(fig_path)
            
            return generated_files
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []


# Default markdown template
DEFAULT_MARKDOWN_TEMPLATE = '''# LLM Red Team Benchmark Report

## Overview

- **Benchmark ID**: {{ benchmark.id }}
- **Name**: {{ benchmark.name }}
- **Version**: {{ benchmark.version }}
- **Date**: {{ benchmark.created_at }}

## Executive Summary

This report presents the results of security testing performed on {{ models|length }} model(s) against a collection of attack vectors designed to assess their resilience to adversarial inputs.

**Overall Success Rate**: {{ "%.2f%%" | format(summary.overall_success_rate * 100) }}

{% if summary.overall_success_rate < benchmark.parameters.success_threshold %}
⚠️ **Warning**: The overall success rate is below the target threshold of {{ "%.2f%%" | format(benchmark.parameters.success_threshold * 100) }}.
{% else %}
✅ The overall success rate meets or exceeds the target threshold of {{ "%.2f%%" | format(benchmark.parameters.success_threshold * 100) }}.
{% endif %}

### Tested Models

{% for model in models %}
- **{{ model.model_id }}** ({{ model.provider }}): {{ "%.2f%%" | format(summary.model_success_rates[model.model_id] * 100) }} success rate
{% endfor %}

### Most Vulnerable Categories

{% for category, rate in most_vulnerable_categories %}
- **{{ category }}**: {{ "%.2f%%" | format(rate * 100) }}% success rate
{% endfor %}

## Test Configuration

- **Sample Size**: {{ benchmark.parameters.sample_size }}
- **Confidence Level**: {{ "%.2f%%" | format(benchmark.parameters.confidence_level * 100) }}
- **Success Threshold**: {{ "%.2f%%" | format(benchmark.parameters.success_threshold * 100) }}
- **Progressive Difficulty**: {{ "Yes" if benchmark.parameters.progressive_difficulty else "No" }}

## Results by Model

| Model | Success Rate | Confidence Interval ({{ "%.0f%%" | format(benchmark.parameters.confidence_level * 100) }}) |
|-------|-------------|-------------|
{% for model in models %}
| {{ model.model_id }} | {{ "%.2f%%" | format(summary.model_success_rates[model.model_id] * 100) }} | {{ "%.2f%%" | format(summary.statistical_analysis.confidence_intervals[model.model_id][0] * 100) }} - {{ "%.2f%%" | format(summary.statistical_analysis.confidence_intervals[model.model_id][1] * 100) }} |
{% endfor %}

## Results by Attack Category

| Category | Success Rate | Samples |
|----------|-------------|---------|
{% for category, examples in examples_by_category.items() %}
| {{ category }} | {{ "%.2f%%" | format(summary.category_success_rates[category] * 100) }} | {{ examples|length }} |
{% endfor %}

## Results by Severity

| Severity | Success Rate |
|----------|-------------|
{% for severity, rate in severity_success_rates.items() %}
| {{ severity }} | {{ "%.2f%%" | format(rate * 100) }} |
{% endfor %}

## Statistical Analysis

### P-Values for Model Comparisons

{% if summary.statistical_analysis.p_values %}
| Comparison | P-Value | Significant Difference |
|------------|---------|------------------------|
{% for comparison, p_value in summary.statistical_analysis.p_values.items() %}
| {{ comparison }} | {{ "%.4f" | format(p_value) if p_value != None else "N/A" }} | {{ "Yes" if p_value != None and p_value < 0.05 else "No" }} |
{% endfor %}
{% else %}
No statistical comparisons available.
{% endif %}

## Recommendations

{% for recommendation in recommendations %}
### {{ recommendation.title }} ({{ recommendation.priority }} priority)

{{ recommendation.description }}

{% endfor %}

## Visualizations

![Model Success Rates](visualizations/model_success_rates.png)

![Category Success Rates](visualizations/category_success_rates.png)

![Severity Heatmap](visualizations/severity_heatmap.png)

---

Report generated on {{ timestamp }} using Red Teamer Framework.
''' 