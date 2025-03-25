import streamlit as st
import pandas as pd
import time
import json
import os
from datetime import datetime
import threading
import queue

# Global queue for communication with the red teaming process
result_queue = queue.Queue()

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'total_prompts' not in st.session_state:
        st.session_state.total_prompts = 0
    if 'completed_prompts' not in st.session_state:
        st.session_state.completed_prompts = 0
    if 'target_model' not in st.session_state:
        st.session_state.target_model = ""
    if 'context_description' not in st.session_state:
        st.session_state.context_description = ""
    if 'context_submitted' not in st.session_state:
        st.session_state.context_submitted = False
    if 'waiting_for_context' not in st.session_state:
        st.session_state.waiting_for_context = True
    if 'force_start' not in st.session_state:
        st.session_state.force_start = False
    if 'debug' not in st.session_state:
        st.session_state.debug = True

def update_results():
    """Update results from the queue"""
    try:
        while not result_queue.empty():
            result = result_queue.get(block=False)
            result_type = result.get('type', '')
            
            if result_type == 'complete':
                st.session_state.is_running = False
                st.rerun()
            elif result_type == 'progress':
                st.session_state.completed_prompts = result.get('completed', 0)
                st.rerun()
            elif result_type == 'result':
                st.session_state.results.append(result)
                st.rerun()
            elif result_type == 'setup':
                st.session_state.target_model = result.get('target_model', "")
                st.session_state.context_description = result.get('context', "")
                st.session_state.total_prompts = result.get('total_prompts', 0)
                st.session_state.start_time = datetime.now()
                st.session_state.is_running = True
                st.session_state.waiting_for_context = False
                st.rerun()
            elif result_type == 'wait_for_context':
                st.session_state.waiting_for_context = True
                st.rerun()
            elif result_type == 'context_received':
                st.session_state.waiting_for_context = False
                st.rerun()
    except queue.Empty:
        pass
    except Exception as e:
        st.error(f"Error updating results: {str(e)}")

def submit_context():
    """Submit the context to the main process"""
    context = st.session_state.context_input
    if not context or context.strip() == "":
        st.error("Please provide a context description before starting")
        return
    
    st.session_state.context_submitted = True
    st.session_state.context_description = context
    st.session_state.waiting_for_context = False
    st.session_state.force_start = True
        
    # Put the context in the result queue - add debugging info
    st.write(f"Submitting context: {context}")
    
    # Clear any existing messages in the queue
    try:
        while not result_queue.empty():
            result_queue.get_nowait()
    except:
        pass
        
    # Send the context submission message
    result_queue.put({
        "type": "context_submitted",
        "context": context
    })
    
    # Show a spinner while waiting for the process to start
    with st.spinner("Starting red teaming process..."):
        # Wait for acknowledgment that context was received
        start_time = time.time()
        context_received = False
        while time.time() - start_time < 10 and not context_received:  # Wait up to 10 seconds
            try:
                if not result_queue.empty():
                    message = result_queue.get(block=False)
                    if message.get('type') == 'context_received':
                        context_received = True
                        st.success("Context received! Starting red teaming...")
                        break
                time.sleep(0.2)
            except queue.Empty:
                time.sleep(0.2)
        
        if not context_received:
            st.warning("Didn't receive confirmation, but proceeding anyway...")
    
    st.experimental_rerun()

def render_context_form():
    """Render the form to collect context information"""
    st.title("🔴 Contextual Red Teaming")
    
    st.markdown("""
    ## Welcome to the Contextual Red Teaming Demo
    
    Please provide a description of your chatbot's purpose and domain to generate tailored adversarial prompts.
    """)
    
    # Example contexts dropdown
    example_contexts = [
        "Please select an example or write your own",
        "A customer service chatbot for an e-commerce platform",
        "A financial advisor chatbot for retirement planning",
        "A healthcare chatbot for patient scheduling and medical advice",
        "A travel booking assistant for vacation planning",
        "A smart home assistant for controlling devices and answering questions"
    ]
    
    selected_example = st.selectbox(
        "Select from example contexts or write your own below:",
        example_contexts
    )
    
    # Default value based on dropdown selection
    default_value = "" if selected_example == example_contexts[0] else selected_example
    
    st.text_area(
        "Describe your chatbot's context, purpose, and domain:",
        value=default_value,
        height=150,
        key="context_input",
        help="For example: 'This is a customer service chatbot for an e-commerce platform that helps with order tracking, returns, and product information.'"
    )
    
    st.button("Start Red Teaming", on_click=submit_context)
    
    st.markdown("""
    ### What happens next?
    
    1. Our system will analyze your chatbot's context
    2. The Dravik red teaming model will generate adversarial prompts tailored to this context
    3. These prompts will be tested against a target model
    4. You'll see real-time results and statistics on this dashboard
    """)

def render_dashboard():
    """Render the main dashboard interface with a chatbot-like experience"""
    st.title("🔴 Real-time Contextual Red Teaming")
    
    # Check if we need to force start
    if st.session_state.force_start and not st.session_state.is_running and not st.session_state.results:
        st.warning("Demo seems stuck. Click to force start:")
        if st.button("Force Start Demo"):
            st.session_state.target_model = "Sample Mode - Gemini 1.5 Pro" 
            st.session_state.total_prompts = 10
            st.session_state.start_time = datetime.now()
            st.session_state.is_running = True
            
            # Import needed modules
            import threading
            import time
            from redteamer.demo.contextual_demo import generate_sample_data
            
            # Function to generate and feed sample data
            def force_start_demo():
                try:
                    # Generate some sample data
                    context = st.session_state.context_description
                    prompts = generate_sample_data(10, context)
                    
                    # Feed data to the UI
                    for i, prompt_data in enumerate(prompts):
                        # Update progress
                        result_queue.put({
                            "type": "progress",
                            "completed": i + 1
                        })
                        
                        # Send result
                        result_queue.put({
                            "type": "result",
                            "prompt": prompt_data["prompt"],
                            "response": prompt_data["response"],
                            "category": prompt_data.get("category", "Unknown"),
                            "successful": prompt_data["successful"],
                            "eval_notes": prompt_data["eval_notes"]
                        })
                        
                        # Small delay
                        time.sleep(1)
                    
                    # Signal completion
                    result_queue.put({
                        "type": "complete"
                    })
                except Exception as e:
                    print(f"Error in force_start_demo: {str(e)}")
            
            # Start the thread
            threading.Thread(target=force_start_demo, daemon=True).start()
            st.experimental_rerun()
    
    # Target information
    st.subheader("Target Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Target Model:** {st.session_state.target_model}")
    with col2:
        elapsed = "N/A"
        if st.session_state.start_time:
            elapsed = str(datetime.now() - st.session_state.start_time).split('.')[0]
        st.markdown(f"**Elapsed Time:** {elapsed}")
    
    # Progress bar
    progress = 0
    if st.session_state.total_prompts > 0:
        progress = st.session_state.completed_prompts / st.session_state.total_prompts
    
    st.progress(progress)
    st.markdown(f"**Progress:** {st.session_state.completed_prompts}/{st.session_state.total_prompts} prompts tested")
    
    # Context description
    with st.expander("Context Description", expanded=True):
        st.text(st.session_state.context_description)
    
    # Results - Present in a chatbot-like conversation interface
    st.subheader("Red Teaming Conversation")
    
    # Container for chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.results:
            st.info("Red teaming in progress... Adversarial prompts will appear here as they're tested.")
            if st.session_state.is_running:
                st.markdown("🤖 **Red Teaming Assistant**: I'm analyzing your chatbot's context and generating targeted attacks...")
        else:
            # Display summary metrics at the top
            successful_attacks = sum(1 for r in st.session_state.results if r.get('successful', False))
            cols = st.columns(3)
            cols[0].metric("Total Tests", st.session_state.completed_prompts)
            cols[1].metric("Successful Attacks", successful_attacks)
            if st.session_state.completed_prompts > 0:
                success_rate = (successful_attacks / st.session_state.completed_prompts) * 100
                cols[2].metric("Success Rate", f"{success_rate:.1f}%")
            
            # Show conversation history in a chat-like interface
            for i, result in enumerate(st.session_state.results):
                # Display the prompt (adversarial user message)
                st.markdown(f"🔴 **Adversarial Prompt ({result.get('category', 'Unknown')}):**")
                st.markdown(f"```\n{result.get('prompt', 'N/A')}\n```")
                
                # Display the response (target model message)
                st.markdown("🤖 **Target Model Response:**")
                # Extract just the response content if it's a dictionary
                response = result.get('response', 'N/A')
                if isinstance(response, dict) and "response" in response:
                    response = response["response"]
                st.markdown(f"```\n{response}\n```")
                
                # Display the evaluation
                status = "✅ Attack Successful" if result.get('successful', False) else "❌ Attack Failed"
                evaluation = f"**Evaluation ({status}):** {result.get('eval_notes', 'N/A')}"
                st.markdown(evaluation)
                
                # Add a separator between conversations
                if i < len(st.session_state.results) - 1:
                    st.markdown("---")
    
    # If still running, show a status message
    if st.session_state.is_running and st.session_state.completed_prompts > 0:
        st.markdown("🔄 Testing in progress...")

def main():
    st.set_page_config(
        page_title="Contextual Red Teaming Dashboard",
        page_icon="🔴",
        layout="wide",
    )
    
    init_session_state()
    
    # Debug info at the top
    if st.session_state.get("debug", False):
        with st.expander("Debug Info"):
            st.write({
                "waiting_for_context": st.session_state.waiting_for_context,
                "context_submitted": st.session_state.context_submitted,
                "is_running": st.session_state.is_running,
                "completed_prompts": st.session_state.completed_prompts,
                "total_prompts": st.session_state.total_prompts,
                "results_count": len(st.session_state.results)
            })
    
    # Process queue messages
    update_results()
    
    # Check if we're waiting for context input
    if st.session_state.waiting_for_context and not st.session_state.context_submitted:
        render_context_form()
    else:
        render_dashboard()
    
    # Auto-refresh
    if st.session_state.is_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main() 