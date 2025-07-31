#!/usr/bin/env python3
"""
NPU Kernel Generation Dashboard

A Streamlit dashboard that allows users to write prompts and generates, 
verifies, and displays traces of NPU kernels.
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
from typing import Dict, Any, Optional

# Import the demo functionality
from npu_kernel_demo import NPUKernelDemo

def init_session_state():
    """Initialize session state variables."""
    if 'demo_results' not in st.session_state:
        st.session_state.demo_results = None
    if 'trace_data' not in st.session_state:
        st.session_state.trace_data = None
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False

def create_trace_visualization(trace_data: dict) -> go.Figure:
    """
    Create a trace visualization from trace data.
    
    Args:
        trace_data: Parsed trace JSON data
        
    Returns:
        Plotly figure with the trace visualization
    """
    try:
        # Use the already parsed trace data
        events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])
        
        # Pair B/E events
        stack = {}
        intervals = []
        for e in events:
            # Skip non-dictionary entries
            if not isinstance(e, dict):
                continue
                
            ph = e.get("ph")
            name = e.get("name")
            tid = e.get("tid", 0)
            ts = e.get("ts", 0)
            
            # Filter out shim trace events - only keep core trace
            process_name = e.get("args", {}).get("name", "")
            if "shim" in str(process_name).lower() or "shim" in str(name).lower():
                continue
                
            if ph == "B":
                stack.setdefault((name, tid), []).append(ts)
            elif ph == "E" and (name, tid) in stack and stack[(name, tid)]:
                start_ts = stack[(name, tid)].pop()
                intervals.append({
                    "start_ns": start_ts,
                    "end_ns": ts,
                    "name": name,
                    "tid": tid,
                    "duration_ns": ts - start_ts
                })

        df = pd.DataFrame(intervals)
        
        if df.empty:
            st.warning("No trace intervals found in the data.")
            return go.Figure()
        
        # Fix negative durations by swapping
        negative_mask = df['duration_ns'] < 0
        if negative_mask.any():
            df.loc[negative_mask, ['start_ns', 'end_ns']] = df.loc[negative_mask, ['end_ns', 'start_ns']].values
            df['duration_ns'] = df['end_ns'] - df['start_ns']
        
        # Normalize to start at 0
        min_start = df["start_ns"].min()
        df["start_ns"] -= min_start
        df["end_ns"] -= min_start
        
        # Separate events into INSTR and DMA categories
        instr_events = df[df['name'].str.startswith('INSTR') | df['name'].str.startswith('PORT') | df['name'].str.startswith('LOCK')]
        dma_events = df[df['name'].str.startswith('DMA')]
        
        instr_names = sorted(instr_events['name'].unique()) if not instr_events.empty else []
        dma_names = sorted(dma_events['name'].unique()) if not dma_events.empty else []
        
        # Create subplots - 2 panels
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Core trace", "Shim trace"),
            row_heights=[len(instr_names), len(dma_names)] if instr_names and dma_names else [1, 1]
        )
        
        # Simple 2-color palette
        colors = ['#4285F4', '#34A853']  # Blue and Green
        
        # Row height similar to Perfetto
        row_height = 0.35
        
        # Add INSTR events to first panel
        for _, row in instr_events.iterrows():
            y_pos = len(instr_names) - instr_names.index(row['name']) - 1
            
            # For very short events, ensure minimum visual width
            duration = row['duration_ns']
            visual_end = row['end_ns']
            if duration <= 1:
                time_range = df['end_ns'].max() - df['start_ns'].min()
                min_visual_width = time_range * 0.0001
                visual_end = row['start_ns'] + max(duration, min_visual_width)
            
            # Use blue for INSTR events
            color = colors[0]
            
            fig.add_trace(go.Scatter(
                x=[row['start_ns'], visual_end, visual_end, row['start_ns'], row['start_ns']],
                y=[y_pos - row_height, y_pos - row_height, y_pos + row_height, y_pos + row_height, y_pos - row_height],
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0.2)', width=0.5),
                name=row['name'],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>" +
                    f"Thread: {row['tid']}<br>" +
                    f"Start: {row['start_ns']:.0f} ns<br>" +
                    f"Duration: {row['duration_ns']:.0f} ns<br>" +
                    f"End: {row['end_ns']:.0f} ns" +
                    "<extra></extra>"
                ),
                mode='lines'
            ), row=1, col=1)
        
        # Add DMA events to second panel
        for _, row in dma_events.iterrows():
            y_pos = len(dma_names) - dma_names.index(row['name']) - 1
            
            # For very short events, ensure minimum visual width
            duration = row['duration_ns']
            visual_end = row['end_ns']
            if duration <= 1:
                time_range = df['end_ns'].max() - df['start_ns'].min()
                min_visual_width = time_range * 0.002
                visual_end = row['start_ns'] + max(duration, min_visual_width)
            
            # Use green for DMA events
            color = colors[1]
            
            fig.add_trace(go.Scatter(
                x=[row['start_ns'], visual_end, visual_end, row['start_ns'], row['start_ns']],
                y=[y_pos - row_height, y_pos - row_height, y_pos + row_height, y_pos + row_height, y_pos - row_height],
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0.2)', width=0.5),
                name=row['name'],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>" +
                    f"Thread: {row['tid']}<br>" +
                    f"Start: {row['start_ns']:.0f} ns<br>" +
                    f"Duration: {row['duration_ns']:.0f} ns<br>" +
                    f"End: {row['end_ns']:.0f} ns" +
                    "<extra></extra>"
                ),
                mode='lines'
            ), row=2, col=1)
        
        # Update layout for both panels - use dark theme to match Streamlit
        fig.update_layout(
            height=max(600, (len(instr_names) + len(dma_names)) * 35),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            font=dict(size=12, color='white'),  # White text
            title="NPU Kernel Execution Trace",
            title_font=dict(color='white')
        )
        
        # Update y-axes for both subplots - dark theme
        if instr_names:
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(len(instr_names))),
                ticktext=list(reversed(instr_names)),
                range=[-0.5, len(instr_names) - 0.5],
                side='left',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',  # Light grid lines
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(255,255,255,0.4)',  # Light axis lines
                tickfont=dict(color='white'),
                row=1, col=1
            )
        
        if dma_names:
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(len(dma_names))),
                ticktext=list(reversed(dma_names)),
                range=[-0.5, len(dma_names) - 0.5],
                side='left',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',  # Light grid lines
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(255,255,255,0.4)',  # Light axis lines
                tickfont=dict(color='white'),
                row=2, col=1
            )
        
        # Style x-axes - dark theme
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.2)',  # Light grid lines
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.4)',  # Light axis lines
            tickfont=dict(color='white')
        )
        
        # Add x-axis title only to bottom panel
        fig.update_xaxes(title_text="Time (ns)", title_font=dict(color='white'), row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trace visualization: {e}")
        return go.Figure()

def run_kernel_generation(prompt: str, kernel_name: str, data_type: str, array_size: int, selected_model: str, status_text=None, progress_bar=None) -> Dict[str, Any]:
    """
    Run the kernel generation pipeline.
    
    Args:
        prompt: User's natural language prompt
        kernel_name: Name for the kernel function
        data_type: Data type for arrays
        array_size: Size of arrays
        
    Returns:
        Dictionary with generation results
    """
    try:
        # Create a local directory for this run
        output_dir = "streamlit_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize demo with selected model and local directory
        demo = NPUKernelDemo(
            model=selected_model, 
            output_dir=output_dir
        )
        
        # Update progress through each step
        if status_text and progress_bar:
            status_text.text("🔧 Compiling kernel...")
            progress_bar.progress(30)
        
        # Run the complete demo pipeline
        result = demo.run_demo(prompt, kernel_name, data_type, array_size)
        
        # Update progress for remaining steps
        if status_text and progress_bar:
            if result.get('success'):
                status_text.text("📦 Generating xclbin...")
                progress_bar.progress(60)
                import time
                time.sleep(0.5)  # Brief pause to show step
                
                status_text.text("🚀 Running kernel on NPU...")
                progress_bar.progress(80)
                time.sleep(0.5)  # Brief pause to show step
                
                status_text.text("📊 Collecting trace data...")
                progress_bar.progress(90)
                time.sleep(0.5)  # Brief pause to show step
        
        # Copy trace data to session if available - look for the actual JSON trace file
        if result.get('success'):
            # Look for JSON trace files in the output directory
            output_dir = "streamlit_results" 
            trace_files = []
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith('.json') and 'trace' in file:
                        trace_files.append(os.path.join(output_dir, file))
            
            if trace_files:
                # Use the first JSON trace file found
                trace_file = trace_files[0]
                try:
                    # Use the exact same logic as the working test plot
                    with open(trace_file) as f:
                        data = json.load(f)
                    events = data if isinstance(data, list) else data.get("traceEvents", [])
                    st.session_state.trace_data = events
                except Exception as e:
                    st.warning(f"Could not read trace file {trace_file}: {e}")
                    st.session_state.trace_data = None
        
        return result
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def main():
    """Main Streamlit app function."""
    st.set_page_config(
        page_title="NPU Kernel Dashboard",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom header with logo
    import base64
    
    # Try to load and encode logo
    try:
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 64px;">'
    except FileNotFoundError:
        logo_html = '<div style="font-size: 48px;">🔧</div>'
    
    st.markdown(f"""
        <div style="background-color: #1f2937; padding: 10px 20px; margin: -1rem -1rem 1rem -1rem; border-bottom: 2px solid #374151;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="color: white; margin: 0; font-size: 20px;">NPU Kernel Generation Dashboard</h1>
                    <p style="color: #9ca3af; margin: 0; font-size: 14px;">Generate, verify, and visualize NPU kernels from natural language prompts</p>
                </div>
                <div>
                    {logo_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_provider = st.selectbox(
            "Model Provider",
            ["OpenAI", "Ollama"],
            help="Choose the LLM provider"
        )
        
        if model_provider == "OpenAI":
            # OpenAI API Key input
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if not api_key:
                api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
                if api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
            else:
                st.success("✅ Using OpenAI API key from environment")
            
            # OpenAI model selection
            openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                help="Choose the OpenAI model"
            )
            selected_model = openai_model
            
        else:  # Ollama
            # Ollama configuration
            ollama_url = st.text_input(
                "Ollama URL", 
                value="http://localhost:11434",
                help="URL of your Ollama server"
            )
            
            ollama_model = st.text_input(
                "Ollama Model",
                value="llama3.1:8b",
                help="Name of the Ollama model to use"
            )
            selected_model = f"ollama:{ollama_model}"
        
        # Set default NPU device if not already set
        if 'NPU' not in os.environ:
            os.environ['NPU'] = 'npu1_1col'
        
        st.info(f"NPU Device: {os.environ.get('NPU', 'npu1_1col')}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard allows you to:
        - Write natural language prompts for NPU kernels
        - Generate C++ kernel code using AI
        - Build and verify kernels on NPU hardware
        - Visualize execution traces
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Preset kernel prompts dropdown (outside form for dynamic updates)
        preset_configs = {
            "Custom (enter your own)": {
                "prompt": "",
                "kernel_name": "my_kernel",
                "data_type": "int8",
                "array_size": 1024
            },
            "ReLU Activation": {
                "prompt": "Write a ReLU kernel that takes in vectors of elements and applies ReLU activation (max(0, x)) to each element",
                "kernel_name": "relu_kernel",
                "data_type": "int8",
                "array_size": 1024
            },
            "Add Offset": {
                "prompt": "Write a kernel that adds a constant offset of 5 to the input array.",
                "kernel_name": "add_offset",
                "data_type": "int16",
                "array_size": 1024
            },
            "Argmax": {
                "prompt": "Return the index of the largest element of the input array.",
                "kernel_name": "argmax",
                "data_type": "int32",
                "array_size": 256
            },
            "Convolution 2D": {
                "prompt": "Write a 2D convolution kernel that applies a filter to an input image",
                "kernel_name": "conv2d_kernel",
                "data_type": "int8",
                "array_size": 512
            },
            "Max Pooling": {
                "prompt": "Write a max pooling kernel that applies 2x2 max pooling to reduce the spatial dimensions",
                "kernel_name": "maxpool_kernel",
                "data_type": "int8",
                "array_size": 1024
            },
            "Sigmoid Activation": {
                "prompt": "Write a sigmoid activation kernel that applies the sigmoid function (1/(1+exp(-x))) to each element",
                "kernel_name": "sigmoid_kernel",
                "data_type": "bfloat16",
                "array_size": 1024
            },
            "Softmax": {
                "prompt": "Write a softmax kernel that computes the softmax function across the last dimension of the input",
                "kernel_name": "softmax_kernel",
                "data_type": "bfloat16",
                "array_size": 1024
            }
        }
        
        preset_choice = st.selectbox(
            "Preset Kernel Prompts",
            options=list(preset_configs.keys()),
            help="Choose a preset prompt or select 'Custom' to enter your own"
        )
        
        # Auto-populate all fields based on selection
        config = preset_configs[preset_choice]
        default_prompt = config["prompt"]
        default_kernel_name = config["kernel_name"]
        default_data_type = config["data_type"]
        default_array_size = config["array_size"]
        
        # Input form
        with st.form("kernel_form"):
            prompt = st.text_area(
                "Kernel Description",
                value=default_prompt,
                placeholder="e.g., Write a ReLU kernel that takes in vectors of 1024 elements of int8",
                height=100,
                help="Describe the kernel operation in natural language"
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                kernel_name = st.text_input(
                    "Kernel Name",
                    value=default_kernel_name,
                    help="Name for the generated kernel function"
                )
            
            with col_b:
                data_type_options = ["int8", "int16", "int32", "bfloat16"]
                # Handle case where default_data_type is not in the new options
                try:
                    data_type_index = data_type_options.index(default_data_type)
                except ValueError:
                    data_type_index = 0  # Default to int8 if unsupported type
                data_type = st.selectbox(
                    "Data Type",
                    data_type_options,
                    index=data_type_index,
                    help="NPU-compatible data types: int8, int16, int32, bfloat16"
                )
            
            with col_c:
                array_size = st.number_input(
                    "Array Size",
                    min_value=256,
                    max_value=4096,
                    value=default_array_size,
                    step=256,
                    help="Size of input/output arrays"
                )
            
            submit_button = st.form_submit_button("🚀 Generate & Verify Kernel", type="primary")
        
        # Generate kernel when form is submitted
        if submit_button:
            if not prompt.strip():
                st.error("Please enter a kernel description")
            elif model_provider == "OpenAI" and not api_key:
                st.error("Please provide an OpenAI API key in the sidebar")
            else:
                with st.spinner("Generating kernel... This may take a few minutes."):
                    # Progress indicators with detailed steps
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🤖 Generating kernel code with AI...")
                    progress_bar.progress(10)
                    
                    # Run generation and update progress as we go
                    result = run_kernel_generation(prompt, kernel_name, data_type, array_size, selected_model, status_text, progress_bar)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Generation complete!")
                    
                    st.session_state.demo_results = result
                    st.session_state.generation_complete = True
                    
                    # Clear progress indicators after a brief delay
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
        
        # Show results underneath the form (after generation logic)
        if st.session_state.demo_results:
            result = st.session_state.demo_results
            
            # Status indicator
            if result.get('success'):
                st.success("✅ Kernel generation and verification successful!")
            else:
                st.error("❌ Kernel generation failed")
                if result.get('error'):
                    st.error(f"Error: {result['error']}")
    
    with col2:
        
        if st.session_state.demo_results:
            result = st.session_state.demo_results
            
            # Generated code - show even if generation failed
            if result.get('generation', {}).get('generated_code'):
                with st.expander("📄 Generated Kernel Code", expanded=True):
                    st.code(result['generation']['generated_code'], language='cpp', height=400)
        
        elif st.session_state.generation_complete:
            st.info("No results to display")
        else:
            st.info("👈 Enter a kernel description and click 'Generate & Verify Kernel' to get started")
    
    # Trace visualization section - always show, populate after generation
    
    if st.session_state.trace_data:
        try:
            fig = create_trace_visualization(st.session_state.trace_data)
            if fig.data:  # Check if figure has data
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trace data available for visualization")
        except Exception as e:
            st.error(f"Error creating trace visualization: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        # Show empty placeholder chart
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="NPU Kernel Execution Trace",
            xaxis_title="Time (ns)",
            yaxis_title="Events",
            height=400,
            annotations=[
                dict(
                    text="Generate a kernel to see trace visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            ]
        )
        st.plotly_chart(empty_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Copyright© 2025 AMD, Inc | SPDX-License-Identifier: MIT"
    )

if __name__ == "__main__":
    main()
