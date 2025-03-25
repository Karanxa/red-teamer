"""
Streamlit-based results viewer for the Red Teaming Framework.

Run this script using:
    streamlit run redteamer/results_viewer.py
"""

import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our Streamlit patch before importing streamlit
from redteamer.red_team.streamlit_patch import safe_import_streamlit

# Import streamlit safely
st = safe_import_streamlit()

# Import visualization libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    """Main entry point for the Streamlit results viewer app."""
    # Initialize the Streamlit UI
    initialize_streamlit_app()
    
    # Find all result files
    result_files = find_result_files()
    
    if not result_files:
        st.error("No result files found. Please run a scan first.")
        st.stop()
    
    # Get selected file from sidebar
    selected_file = sidebar_file_selector(result_files)
    
    # Display the results
    try:
        with open(selected_file, 'r') as f:
            results = json.load(f)
        
        display_results(results, selected_file)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)

def initialize_streamlit_app():
    """Initialize the Streamlit app with basic configuration."""
    st.set_page_config(
        page_title="LLM Red Teaming - Results Viewer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 LLM Red Teaming Results Viewer")
    st.markdown("---")

def find_result_files() -> List[str]:
    """Find all result files in the results directory."""
    # Look for JSON files in the results directory and subdirectories
    return glob.glob("results/**/*.json", recursive=True)

def sidebar_file_selector(result_files: List[str]) -> str:
    """Display a sidebar to select result files."""
    st.sidebar.header("Select Results File")
    
    # Parse metadata from result files
    file_metadata = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            metadata = data.get('metadata', {})
            name = metadata.get('name', 'Unknown')
            benchmark_id = metadata.get('benchmark_id', os.path.basename(file_path))
            created_at = metadata.get('created_at', '')
            
            if created_at:
                try:
                    # Format timestamp for display
                    dt = datetime.fromisoformat(created_at)
                    timestamp = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    timestamp = created_at
            else:
                timestamp = 'Unknown'
            
            # Check scan type
            is_conversation = 'conversation' in file_path.lower() or 'conv_redteam' in file_path.lower()
            is_static = 'static' in file_path.lower()
            
            scan_type = "Conversation Scan" if is_conversation else "Static Scan" if is_static else "Other Scan"
            
            # Store metadata for display
            file_metadata.append({
                'path': file_path,
                'name': name,
                'benchmark_id': benchmark_id,
                'timestamp': timestamp,
                'scan_type': scan_type
            })
        except Exception as e:
            # If we can't parse the file, just use the filename
            file_metadata.append({
                'path': file_path,
                'name': os.path.basename(file_path),
                'benchmark_id': os.path.basename(file_path),
                'timestamp': 'Unknown',
                'scan_type': 'Unknown'
            })
    
    # Sort by timestamp, newest first
    file_metadata.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Group by scan type
    scan_types = set(item['scan_type'] for item in file_metadata)
    
    # Create a select box for scan type filter
    selected_type = st.sidebar.selectbox(
        "Filter by Scan Type",
        ["All Types"] + list(scan_types)
    )
    
    # Filter by selected type
    if selected_type != "All Types":
        file_metadata = [item for item in file_metadata if item['scan_type'] == selected_type]
    
    if not file_metadata:
        st.sidebar.warning("No results found with the selected filter.")
        # Return the first file as fallback
        return result_files[0] if result_files else ""
    
    # Create a radio button selection for files
    options = [f"{item['name']} ({item['timestamp']})" for item in file_metadata]
    selected_option = st.sidebar.radio("Select Results", options)
    
    # Get the index of the selected option
    selected_index = options.index(selected_option)
    
    # Get the selected file path
    selected_file_path = file_metadata[selected_index]['path']
    
    # Add a "Delete this result" button in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Result Management")
    
    # Add delete button with confirmation
    delete_clicked = st.sidebar.button("Delete Selected Result", type="primary", use_container_width=True)
    
    if delete_clicked:
        delete_confirmed = st.sidebar.checkbox("I understand this action cannot be undone. Confirm deletion.", value=False)
        
        if delete_confirmed:
            try:
                # Delete the selected file
                os.remove(selected_file_path)
                
                # Check for associated report files
                report_base = os.path.basename(selected_file_path).replace('.json', '_report')
                report_dir = os.path.join('reports')
                
                # Try to delete any associated report files
                if os.path.exists(report_dir):
                    for ext in ['.markdown', '.md', '.json', '.csv', '.pdf']:
                        report_path = os.path.join(report_dir, f"{report_base}{ext}")
                        if os.path.exists(report_path):
                            os.remove(report_path)
                
                # Show success message
                st.sidebar.success(f"Result deleted successfully. Please refresh the page.")
                
                # Force page reload after 2 seconds
                st.sidebar.markdown("""
                <script>
                    setTimeout(function() {
                        window.location.reload();
                    }, 2000);
                </script>
                """, unsafe_allow_html=True)
                
                # Return a different file if available
                remaining_files = [f for f in result_files if f != selected_file_path]
                if remaining_files:
                    return remaining_files[0]
                else:
                    st.sidebar.warning("No more results available.")
                    st.stop()
            except Exception as e:
                st.sidebar.error(f"Error deleting file: {str(e)}")
    
    # Return the selected file path
    return selected_file_path

def display_results(results: Dict[str, Any], file_path: str):
    """Display the results from the selected file."""
    # Determine scan type
    is_conversation = 'conversation' in file_path.lower() or 'conv_redteam' in file_path.lower()
    is_static = 'static' in file_path.lower()
    
    # Get metadata
    metadata = results.get('metadata', {})
    
    # Add actions to the top of the page
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Display JSON download button
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=os.path.basename(file_path),
            mime="application/json"
        )
    
    with col2:
        # Try to find an associated report file
        try:
            report_basename = os.path.basename(file_path).replace('.json', '_report.markdown')
            report_path = os.path.join('reports', report_basename)
            
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_content = f.read()
                    st.download_button(
                        label="Download Report",
                        data=report_content,
                        file_name=report_basename,
                        mime="text/markdown"
                    )
        except:
            pass
    
    with col3:
        # Generate report button
        if st.button("Generate New Report", use_container_width=True):
            generate_report(file_path)
    
    # Display a horizontal rule for separation
    st.markdown("---")
    
    # Display different views based on scan type
    if is_conversation:
        display_conversation_results(results)
    else:
        display_static_results(results)

def display_conversation_results(results: Dict[str, Any]):
    """Display results from a conversation scan."""
    st.header("Conversation Red Teaming Results")
    
    # Get metadata and summary
    metadata = results.get('metadata', {})
    summary = results.get('summary', {})
    
    # Display scan info
    st.subheader("Scan Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Model", results.get('target_model', 'Unknown'))
    
    with col2:
        st.metric("Red Team Model", results.get('redteam_model', 'Unknown'))
    
    with col3:
        st.metric("Total Iterations", summary.get('total_iterations', 0))
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Vulnerabilities Found", summary.get('total_vulnerabilities', 0))
    
    with col2:
        vulnerability_rate = summary.get('vulnerability_rate', 0) * 100
        st.metric("Vulnerability Rate", f"{vulnerability_rate:.1f}%")
    
    with col3:
        # Calculate duration in minutes
        start_time = results.get('start_time', 0)
        end_time = results.get('end_time', 0)
        if start_time and end_time:
            duration_minutes = (end_time - start_time) / 60
            st.metric("Duration", f"{duration_minutes:.1f} minutes")
    
    # Display classification summary with visualization
    classifications = summary.get('classifications', {})
    if classifications:
        st.subheader("Response Classifications")
        
        # Create a DataFrame for the pie chart
        df = pd.DataFrame({
            'Classification': list(classifications.keys()),
            'Count': list(classifications.values())
        })
        
        # Create two columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display counts in a table
            st.write(df)
        
        with col2:
            # Create a pie chart
            fig = px.pie(
                df, 
                values='Count', 
                names='Classification',
                title='Distribution of Response Classifications',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display vulnerabilities
    vulnerabilities = results.get('vulnerabilities', [])
    if vulnerabilities:
        st.subheader("Detected Vulnerabilities")
        
        for i, vuln in enumerate(vulnerabilities):
            with st.expander(f"Vulnerability #{i+1} - {vuln.get('classification', 'Unknown')}"):
                st.markdown("**Prompt:**")
                st.code(vuln.get('prompt', ''))
                
                st.markdown("**Response:**")
                # Extract just the response content if it's a dictionary
                response = vuln.get('response', '')
                if isinstance(response, dict) and "response" in response:
                    response = response["response"]
                st.code(response)
                
                # Display classification details if available
                st.markdown(f"**Classification:** {vuln.get('classification', 'Unknown')}")
    
    # Display full conversation history
    conversation = results.get('conversation', [])
    if conversation:
        st.subheader("Complete Conversation History")
        
        for i, entry in enumerate(conversation):
            with st.expander(f"Iteration {entry.get('iteration', i+1)}", expanded=False):
                st.markdown("**Prompt:**")
                st.text(entry.get('prompt', ''))
                
                st.markdown("**Response:**")
                # Extract just the response content if it's a dictionary
                response = entry.get('response', '')
                if isinstance(response, dict) and "response" in response:
                    response = response["response"]
                st.text(response)
                
                # Display analysis
                analysis = entry.get('analysis', {})
                if analysis:
                    st.markdown("**Analysis:**")
                    classification = analysis.get('classification', 'Unknown')
                    st.markdown(f"Classification: {classification}")
                    st.markdown(f"Potential Vulnerability: {analysis.get('potential_vulnerability', False)}")

def display_static_results(results: Dict[str, Any]):
    """Display results from a static scan."""
    st.header("Static Scan Results")
    
    # Get metadata and summary
    metadata = results.get('metadata', {})
    summary = results.get('summary', {})
    
    # Get all results
    all_results = results.get('results', [])
    
    # Extract key information for the overview
    total_vectors = summary.get('total_tests', len(all_results))
    vulnerability_count = summary.get('vulnerability_count', sum(1 for r in all_results if r.get('success', False)))
    
    # Calculate vulnerability rate
    vulnerability_rate = summary.get('vulnerability_rate', vulnerability_count / total_vectors if total_vectors > 0 else 0)
    
    # Display overview metrics
    st.subheader("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vectors", total_vectors)
    
    with col2:
        st.metric("Vulnerabilities Found", vulnerability_count)
    
    with col3:
        st.metric("Vulnerability Rate", f"{vulnerability_rate*100:.1f}%")
        
    with col4:
        models_tested = len(summary.get("models", []))
        st.metric("Models Tested", models_tested)
    
    # Display scan details
    st.subheader("Scan Details")
    
    scan_info = {
        "Name": metadata.get("name", "Unknown"),
        "Benchmark ID": metadata.get("benchmark_id", "Unknown"),
        "Date": metadata.get("created_at", "Unknown"),
        "Description": metadata.get("description", "")
    }
    
    # Display model details from summary
    st.markdown("### Model Details")
    
    models_summary = summary.get("models", [])
    if models_summary:
        model_data = []
        
        for model_info in models_summary:
            model_name = model_info.get("model", "Unknown")
            vuln_rate = model_info.get("vulnerability_rate", 0.0)
            total_tests = model_info.get("total_tests", 0)
            vuln_count = model_info.get("vulnerable_count", 0)
            
            model_data.append({
                "Model": model_name,
                "Total Tests": total_tests,
                "Vulnerabilities": vuln_count,
                "Vulnerability Rate": f"{vuln_rate*100:.1f}%"
            })
        
        # Create a DataFrame for the models
        if model_data:
            models_df = pd.DataFrame(model_data)
            st.dataframe(models_df)
            
            # Create a bar chart for vulnerability rates
            fig = px.bar(
                models_df,
                x="Model", 
                y=[float(rate.strip('%')) for rate in models_df["Vulnerability Rate"]],
                labels={"y": "Vulnerability Rate (%)", "x": "Model"},
                title="Vulnerability Rate by Model",
                color=[float(rate.strip('%')) for rate in models_df["Vulnerability Rate"]],
                color_continuous_scale="RdYlGn_r"
            )
            
            st.plotly_chart(fig)
    
    # Display attack vector details
    st.markdown("### Attack Vector Details")
    
    # Prepare data for the table
    vectors_data = []
    
    for result in all_results:
        success = result.get("success", False)
        confidence = result.get("confidence", 0.0)
        vector_id = result.get("vector_id", "Unknown")
        
        # Get prompt and response
        prompt = result.get("prompt", "")
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        # Extract just the response content
        response = result.get("response", "")
        if isinstance(response, dict) and "response" in response:
            response = response["response"]
        response_preview = response[:100] + "..." if len(response) > 100 else response
        
        # Get explanation
        explanation = result.get("explanation", "No explanation provided")
        
        # Add to data
        vectors_data.append({
            "Vector ID": vector_id,
            "Attack Success": "🔴 Yes" if success else "🟢 No",
            "Confidence": f"{confidence*100:.1f}%",
            "Prompt Preview": prompt_preview,
            "Response Preview": response_preview,
            "Original Data": result  # Store original data for expanded view
        })
    
    # Create DataFrame for the table
    if vectors_data:
        vectors_df = pd.DataFrame(vectors_data)
        
        # Display summary statistics
        st.subheader("Attack Results Overview")
        
        # Count successes and failures
        success_count = sum(1 for r in vectors_data if "Yes" in r["Attack Success"])
        failure_count = sum(1 for r in vectors_data if "No" in r["Attack Success"])
        
        # Create a pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Successful Attacks", "Failed Attacks"],
                    values=[success_count, failure_count],
                    hole=0.4,
                    marker=dict(colors=["#FF5252", "#4CAF50"])
                )
            ]
        )
        fig.update_layout(title_text="Attack Success Rate")
        
        st.plotly_chart(fig)
        
        # Display the table
        st.subheader("Individual Attack Vectors")
        
        # Add a filter for attack success/failure
        filter_options = ["All Vectors", "Successful Attacks Only", "Failed Attacks Only"]
        selected_filter = st.selectbox("Filter attacks by result:", filter_options)
        
        if selected_filter == "Successful Attacks Only":
            filtered_df = vectors_df[vectors_df["Attack Success"].str.contains("Yes")]
        elif selected_filter == "Failed Attacks Only":
            filtered_df = vectors_df[vectors_df["Attack Success"].str.contains("No")]
        else:
            filtered_df = vectors_df
        
        # Sort by attack success (descending) and confidence (descending)
        filtered_df = filtered_df.sort_values(
            by=["Attack Success", "Confidence"],
            ascending=[False, False]
        )
        
        # Drop the original data column for display
        display_df = filtered_df.drop(columns=["Original Data"])
        
        # Show the table
        st.dataframe(display_df)
        
        # Add detailed views for each vector
        st.subheader("Detailed Attack Vector Analysis")
        
        for i, row in filtered_df.iterrows():
            original_data = row["Original Data"]
            vector_id = row["Vector ID"]
            success = "Yes" if original_data.get("success", False) else "No"
            success_emoji = "🔴" if success == "Yes" else "🟢"
            
            # Create an expander for each vector
            with st.expander(f"Vector {vector_id} - Attack Success: {success_emoji} {success}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Prompt")
                    st.text_area("", original_data.get("prompt", ""), height=150, key=f"prompt_{vector_id}")
                
                with col2:
                    st.markdown("#### Response")
                    # Extract just the response content
                    response = original_data.get("response", "")
                    if isinstance(response, dict) and "response" in response:
                        response = response["response"]
                    st.text_area("", response, height=150, key=f"response_{vector_id}")
                
                # Display evaluation details
                st.markdown("#### Evaluation Details")
                
                eval_col1, eval_col2 = st.columns(2)
                
                with eval_col1:
                    confidence_value = original_data.get('confidence', 0.0) * 100
                    confidence_color = "red" if confidence_value > 80 else "orange" if confidence_value > 50 else "green"
                    
                    st.markdown(f"**Success:** {success_emoji} {success}")
                    st.markdown(f"**Confidence:** <span style='color:{confidence_color};font-weight:bold;'>{confidence_value:.1f}%</span>", unsafe_allow_html=True)
                    
                    # Add severity if available
                    if "severity" in original_data:
                        severity = original_data.get("severity", "Unknown")
                        severity_color = "red" if severity.lower() == "high" else "orange" if severity.lower() == "medium" else "blue"
                        st.markdown(f"**Severity:** <span style='color:{severity_color};font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
                
                with eval_col2:
                    matched_rules = original_data.get("matched_rules", [])
                    if matched_rules:
                        st.markdown("**Matched Rules:**")
                        
                        # Categorize rules by type
                        attack_rules = []
                        defense_rules = []
                        other_rules = []
                        
                        for rule in matched_rules:
                            if "refusal" in rule or "safety" in rule or "defense" in rule:
                                defense_rules.append(rule)
                            elif "attack" in rule or "harmful" in rule or "jailbreak" in rule or "exploit" in rule:
                                attack_rules.append(rule)
                            else:
                                other_rules.append(rule)
                        
                        # Display rules with color coding
                        if attack_rules:
                            st.markdown("*Attack signals:*")
                            for rule in attack_rules:
                                st.markdown(f"<span style='color:red;'>- {rule}</span>", unsafe_allow_html=True)
                        
                        if defense_rules:
                            st.markdown("*Defense signals:*")
                            for rule in defense_rules:
                                st.markdown(f"<span style='color:green;'>- {rule}</span>", unsafe_allow_html=True)
                        
                        if other_rules:
                            st.markdown("*Other signals:*")
                            for rule in other_rules:
                                st.markdown(f"- {rule}")
                    else:
                        st.markdown("**Matched Rules:** None")
                
                # Display explanation with better formatting
                st.markdown("#### Evaluation Explanation")
                explanation_box_color = "rgba(255, 82, 82, 0.1)" if success == "Yes" else "rgba(76, 175, 80, 0.1)"
                explanation = original_data.get("explanation", "No explanation provided")
                
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {explanation_box_color};">
                        {explanation}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Additional metadata in a clean format
                st.markdown("#### Vector Metadata")
                
                metadata_col1, metadata_col2 = st.columns(2)
                
                with metadata_col1:
                    # Display category and other attack vector specific info
                    if "category" in original_data:
                        st.markdown(f"**Category:** {original_data.get('category', 'Unknown')}")
                    
                    if "target_capability" in original_data:
                        st.markdown(f"**Target Capability:** {original_data.get('target_capability', 'Unknown')}")
                
                with metadata_col2:
                    # Display model specific info
                    if "model_id" in original_data:
                        st.markdown(f"**Model:** {original_data.get('provider', 'Unknown')}/{original_data.get('model_id', 'Unknown')}")
                    
                    if "vector_id" in original_data:
                        st.markdown(f"**Vector ID:** {original_data.get('vector_id', 'Unknown')}")
    else:
        st.warning("No attack vectors found in the results.")
        
    # Display raw JSON option
    with st.expander("View Raw JSON Data"):
        st.json(results)

def generate_report(file_path: str):
    """Generate a report from the results file."""
    try:
        # Generate report using subprocess
        import subprocess
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Default output path
        output_path = os.path.join("reports", os.path.basename(file_path).replace('.json', '_report.markdown'))
        
        # Run the CLI report generation command
        process = subprocess.Popen(
            ["python", "-m", "redteamer.cli", "report", "generate", file_path, "--output", output_path, "--format", "markdown", "--non-interactive"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            st.success(f"Report generated successfully: {output_path}")
            
            # Add a download button for the generated report
            try:
                with open(output_path, 'r') as f:
                    report_content = f.read()
                    st.download_button(
                        label="Download Generated Report",
                        data=report_content,
                        file_name=os.path.basename(output_path),
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"Error reading generated report: {str(e)}")
        else:
            st.error(f"Error generating report: {stderr.decode()}")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main() 