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
        # Basic benchmark information
        benchmark_info = {
            "id": self.results["metadata"]["benchmark_id"],
            "name": self.results["metadata"]["name"],
            "description": self.results["metadata"]["description"],
            "version": self.results["metadata"]["version"],
            "created_at": self.results["metadata"].get("created_at", datetime.datetime.now().isoformat()),
            "parameters": self.results["metadata"]["parameters"]
        }
        
        # Model information
        models = self.results["models_tested"]
        
        # Summary statistics
        summary = self.results["summary"]
        
        # Examples (test cases)
        examples = self.results["examples"]
        
        # Group examples by category
        examples_by_category = {}
        for example in examples:
            category = example["evaluation"]["attack_category"]
            if category not in examples_by_category:
                examples_by_category[category] = []
            examples_by_category[category].append(example)
        
        # Calculate additional statistics
        
        # Success rate by severity
        severity_success_rates = {}
        severity_counts = {}
        
        for example in examples:
            severity = example["evaluation"]["severity"]
            
            if severity not in severity_success_rates:
                severity_success_rates[severity] = 0
                severity_counts[severity] = 0
            
            for response in example["responses"]:
                success = response.get("success", False)
                severity_success_rates[severity] += 1 if success else 0
                severity_counts[severity] += 1
        
        for severity, count in severity_counts.items():
            if count > 0:
                severity_success_rates[severity] = severity_success_rates[severity] / count
        
        # Calculate most vulnerable categories
        categories = [(cat, rate) for cat, rate in summary["category_success_rates"].items()]
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
        
        # Look at the most vulnerable categories
        categories = [(cat, rate) for cat, rate in self.results["summary"]["category_success_rates"].items()]
        categories.sort(key=lambda x: x[1])
        
        # Add recommendations for the lowest performing categories
        for category, rate in categories[:3]:
            if rate < self.results["metadata"]["parameters"]["success_threshold"]:
                recommendations.append({
                    "category": "Vulnerability",
                    "title": f"Improve {category} defenses",
                    "description": f"Models demonstrated significant vulnerability to {category} attacks (success rate: {rate:.2%}). "
                                  f"Review and strengthen defenses against this attack vector.",
                    "priority": "high" if rate < 0.3 else "medium"
                })
        
        # Look at model performance differences
        model_rates = self.results["summary"]["model_success_rates"]
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
        overall_rate = self.results["summary"]["overall_success_rate"]
        threshold = self.results["metadata"]["parameters"]["success_threshold"]
        
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
            
            # 1. Model success rates
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
            
            # 3. Severity heatmap
            plt.figure(figsize=(10, 8))
            
            # Create a DataFrame from examples
            data = []
            for example in self.results["examples"]:
                category = example["evaluation"]["attack_category"]
                severity = example["evaluation"]["severity"]
                
                for response in example["responses"]:
                    model = response["model_id"]
                    success = 1 if response.get("success", False) else 0
                    
                    data.append({
                        "model": model,
                        "category": category,
                        "severity": severity,
                        "success": success
                    })
            
            df = pd.DataFrame(data)
            
            # Create pivot table for heatmap
            pivot = df.pivot_table(
                index="category", 
                columns="severity", 
                values="success", 
                aggfunc="mean"
            ).fillna(0)
            
            # Sort index by average success rate
            pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
            
            # Get severity order
            severity_order = ["low", "medium", "high", "critical"]
            severity_columns = [s for s in severity_order if s in pivot.columns]
            pivot = pivot[severity_columns]
            
            # Plot
            ax = sns.heatmap(
                pivot, 
                annot=True, 
                fmt=".2%", 
                cmap="RdYlGn_r",
                vmin=0, 
                vmax=1,
                linewidths=.5
            )
            ax.set_title("Success Rate by Category and Severity")
            
            # Save figure
            fig_path = os.path.join(vis_dir, "severity_heatmap.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            generated_files.append(fig_path)
            
            return generated_files
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
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