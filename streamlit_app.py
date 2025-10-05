import streamlit as st
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from agent_rag import (
        DocumentProcessor, 
        HybridRetriever, 
        OrchestrationAgent,  
        agentic_rag_system  
    )
    
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.error("Please ensure agent_rag.py is in the same directory and all dependencies are installed.")
    st.stop()

st.set_page_config(
    page_title="Agentic RAG System Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .comparison-card {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2196f3;
        margin: 1rem 0;
    }
    .success-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agentic_rag_system():
    try:
        with st.spinner("🔄 Loading Agentic RAG System..."):
            pdf_path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found at {pdf_path}")
                st.error("Please ensure the PDF file is in the correct location.")
                return None

            system = agentic_rag_system(
                pdf_path=pdf_path,
                chunk_size=800,
                chunk_overlap=100,
                use_hf_llm=True
            )
            
            class AgenticRAGWrapper:
                def __init__(self, system):
                    self.orchestrator = system['orchestrator']  
                    self.retriever = system['retriever']
                    self.processor = system['processor']
                    self.documents = system['documents']
                    self.chunks = system['chunks']
                
                def execute(self, query, use_traditional=False):
                    if use_traditional:
                        response = self.orchestrator.execute_traditional(query)
                    else:
                        response = self.orchestrator.execute(query)
                    
                    return response
                
                def get_system_stats(self):
                    stats = self.orchestrator.get_system_stats()
                    return stats
            
            return AgenticRAGWrapper(system)
            
    except Exception as e:
        st.error(f"Failed to load Agentic RAG system: {e}")
        st.error(f"Error details: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic RAG System Demo</h1>
        <p>Intelligent Document Retrieval with Multi-Agent Reasoning vs Traditional RAG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load system
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = load_agentic_rag_system()
        if st.session_state.orchestrator is None:
            st.error("Failed to load the Agentic RAG system. Please check your setup.")
            st.stop()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # RAG Mode Selection
        st.subheader("🔄 RAG Mode")
        rag_mode = st.radio(
            "Select RAG approach:",
            ["Agentic RAG", "Traditional RAG", "Compare Both"],
            help="Choose between agentic approach with multi-agent coordination or traditional single-shot retrieval"
        )
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            quality_threshold = st.slider("Quality Threshold", 0.1, 1.0, 0.6, 0.1)
            max_iterations = st.slider("Max Iterations (Agentic only)", 1, 5, 3)
            
            st.subheader("Retrieval Strategy Weights")
            semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.1)
            lexical_weight = st.slider("Lexical Weight", 0.0, 1.0, 0.4, 0.1)
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.orchestrator:
            stats = st.session_state.orchestrator.get_system_stats()
            st.metric("Total Conversations", stats.get('total_conversations', 0))
            st.metric("Avg Quality", f"{stats.get('average_quality_score', 0):.3f}")
            st.metric("Avg Response Time", f"{stats.get('average_response_time', 0):.2f}s")
        
        # Document Collection Info
        st.subheader("📚 Document Collection")
        st.info("📄 Attention Is All You Need Paper\n15 pages")
    
    # Main Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Query Interface")
        
        # Sample queries
        sample_queries = [
            "Compare scaled dot-product attention with additive attention",
            "How does the transformer handle long sequences?",
            "What is the role of layer normalization in transformers?"
        ]
        
        selected_sample = st.selectbox(
            "🎯 Sample Queries (or enter your own below):",
            [""] + sample_queries
        )
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            value=selected_sample,
            height=100,
            placeholder="Ask anything about transformer architectures, attention mechanisms, or neural networks..."
        )
        
        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col_b:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        with col_c:
            compare_button = st.button("⚖️ Compare", use_container_width=True, disabled=(rag_mode != "Compare Both"))
        
        if clear_button:
            if 'search_history' in st.session_state:
                st.session_state.search_history = []
            if 'comparison_results' in st.session_state:
                st.session_state.comparison_results = []
            st.rerun()
    
    with col2:
        st.header("🎯 Quick Actions")
        
        # Mode indicator
        mode_color = {"Agentic RAG": "🟢", "Traditional RAG": "🔵", "Compare Both": "🟣"}
        st.markdown(f"""
        <div class="metric-card">
            <strong>Current Mode:</strong><br>
            {mode_color.get(rag_mode, "🔴")} {rag_mode}
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        if 'search_history' in st.session_state and st.session_state.search_history:
            latest = st.session_state.search_history[-1]
            st.markdown(f"""
            <div class="metric-card">
                <strong>Last Query Results:</strong><br>
                Quality: {latest['quality_metrics']['overall_score']:.3f}<br>
                Time: {latest['process_metadata']['total_time']:.2f}s<br>
                Approach: {latest['process_metadata']['approach'].title()}
            </div>
            """, unsafe_allow_html=True)
    
    # Search Execution
    if (search_button or compare_button) and query.strip():
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []
        
        if compare_button or rag_mode == "Compare Both":
            # Compare both approaches
            st.markdown("### ⚖️ **Comparison Mode: Traditional vs Agentic RAG**")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown("#### 🔵 Traditional RAG")
                with st.spinner("Running Traditional RAG..."):
                    trad_response = st.session_state.orchestrator.execute(query, use_traditional=True)
                
                st.markdown(f"""
                <div class="comparison-card">
                    <strong>Quality Score:</strong> {trad_response['quality_metrics']['overall_score']:.3f}<br>
                    <strong>Response Time:</strong> {trad_response['process_metadata']['total_time']:.2f}s<br>
                    <strong>Iterations:</strong> {trad_response['process_metadata']['iterations_used']}<br>
                    <strong>Strategies:</strong> {', '.join(trad_response['process_metadata']['strategies_tried'])}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Traditional RAG Response"):
                    st.write(trad_response['response'])
            
            with col_comp2:
                st.markdown("#### 🟢 Agentic RAG")
                with st.spinner("Running Agentic RAG..."):
                    agent_response = st.session_state.orchestrator.execute(query, use_traditional=False)
                
                st.markdown(f"""
                <div class="comparison-card">
                    <strong>Quality Score:</strong> {agent_response['quality_metrics']['overall_score']:.3f}<br>
                    <strong>Response Time:</strong> {agent_response['process_metadata']['total_time']:.2f}s<br>
                    <strong>Iterations:</strong> {agent_response['process_metadata']['iterations_used']}<br>
                    <strong>Strategies:</strong> {', '.join(agent_response['process_metadata']['strategies_tried'])}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Agentic RAG Response"):
                    st.write(agent_response['response'])
            
            # Comparison Analysis
            quality_diff = agent_response['quality_metrics']['overall_score'] - trad_response['quality_metrics']['overall_score']
            time_diff = agent_response['process_metadata']['total_time'] - trad_response['process_metadata']['total_time']
            
            st.markdown("#### 📊 **Comparison Results**")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric("Quality Improvement", f"{quality_diff:+.3f}", delta=f"{quality_diff:+.3f}")
            
            with comp_col2:
                st.metric("Time Overhead", f"+{time_diff:.2f}s", delta=f"+{time_diff:.2f}s")
            
            with comp_col3:
                iteration_diff = agent_response['process_metadata']['iterations_used'] - trad_response['process_metadata']['iterations_used']
                st.metric("Extra Iterations", f"+{iteration_diff}", delta=f"+{iteration_diff}")
            
            # Winner determination
            if quality_diff > 0.05:
                st.markdown("""
                <div class="success-card">
                    🏆 <strong>Winner: Agentic RAG</strong><br>
                    Significant quality improvement through multi-agent coordination and iterative refinement.
                </div>
                """, unsafe_allow_html=True)
            elif quality_diff > 0:
                st.markdown("""
                <div class="warning-card">
                    ✅ <strong>Winner: Agentic RAG (Marginal)</strong><br>
                    Modest quality improvement with additional processing time.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <strong>Winner: Traditional RAG</strong><br>
                    Traditional approach achieved similar or better results with less computation.
                </div>
                """, unsafe_allow_html=True)
            
            # Store comparison
            comparison_result = {
                'query': query,
                'traditional': trad_response,
                'agentic': agent_response,
                'quality_improvement': quality_diff,
                'time_overhead': time_diff,
                'timestamp': time.time()
            }
            st.session_state.comparison_results.append(comparison_result)
            
        else:
            # Single mode execution
            use_traditional = (rag_mode == "Traditional RAG")
            approach_name = "Traditional RAG" if use_traditional else "Agentic RAG"
            
            with st.spinner(f"🤖 {approach_name} agents are working..."):
                response = st.session_state.orchestrator.execute(query, use_traditional=use_traditional)
            
            st.success(f"✅ {approach_name} search completed!")
            st.session_state.search_history.append(response)
    
    # Display Results
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.header("📋 Search Results")
        
        # Show latest result
        latest_result = st.session_state.search_history[-1]
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality Score", f"{latest_result['quality_metrics']['overall_score']:.3f}")
        with col2:
            st.metric("Confidence", f"{latest_result['quality_metrics'].get('confidence', 0.0):.3f}")
        with col3:
            st.metric("Iterations", latest_result['process_metadata']['iterations_used'])
        with col4:
            st.metric("Response Time", f"{latest_result['process_metadata']['total_time']:.2f}s")
        
        # Content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Response", "🔗 Citations", "📊 Process", "🔍 Raw Results", "📈 Analytics"])
        
        with tab1:
            st.subheader("Generated Response")
            st.markdown(f"**Approach:** {latest_result['process_metadata']['approach'].title()}")
            st.markdown(f"**Generation Method:** {latest_result.get('generation_method', 'N/A')}")
            st.write(latest_result['response'])
        
        with tab2:
            st.subheader("Citations & Sources")
            for i, citation in enumerate(latest_result['citations'], 1):
                st.write(f"{i}. {citation}")
        
        with tab3:
            st.subheader("Agent Process Flow")
            
            # Process visualization
            if latest_result['process_metadata']['approach'] == 'agentic':
                process_data = {
                    'Step': ['Query Analysis', 'Retrieval 1', 'Quality Assessment', 'Retrieval 2', 'Citation Generation'],
                    'Agent': ['Query Analyzer', 'Retrieval Agent', 'Quality Assessor', 'Retrieval Agent', 'Citation Agent'],
                    'Action': ['Analyze query type', 'Initial search', 'Assess quality', 'Refined search', 'Generate citations'],
                    'Status': ['✅ Completed'] * 5
                }
            else:
                process_data = {
                    'Step': ['Direct Retrieval', 'Citation Generation'],
                    'Agent': ['Retrieval Agent', 'Citation Agent'],
                    'Action': ['Semantic search', 'Generate citations'],
                    'Status': ['✅ Completed'] * 2
                }
            
            df = pd.DataFrame(process_data)
            st.dataframe(df, use_container_width=True)
            
            # Strategy progression
            strategies = latest_result['process_metadata']['strategies_tried']
            st.write(f"**Strategies Used:** {' → '.join(strategies)}")
        
        with tab4:
            st.subheader("Retrieved Documents")
            for i, result in enumerate(latest_result['results'][:3], 1):
                with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                    st.write(f"**Page:** {result['page_number']}")
                    st.write(f"**Method:** {result['retrieval_method']}")
                    st.write(f"**Content:** {result['content']}")
        
        with tab5:
            st.subheader("Performance Analytics")
            
            if len(st.session_state.search_history) > 1:
                # Create performance chart
                history_df = pd.DataFrame([
                    {
                        'Query': i+1,
                        'Quality Score': result['quality_metrics']['overall_score'],
                        'Response Time': result['process_metadata']['total_time'],
                        'Iterations': result['process_metadata']['iterations_used'],
                        'Approach': result['process_metadata']['approach']
                    }
                    for i, result in enumerate(st.session_state.search_history)
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.line(history_df, x='Query', y='Quality Score', 
                                 color='Approach', title='Quality Score Trend', markers=True)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.line(history_df, x='Query', y='Response Time',
                                 color='Approach', title='Response Time Trend', markers=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                summary_stats = history_df.groupby('Approach').agg({
                    'Quality Score': ['mean', 'std'],
                    'Response Time': ['mean', 'std'],
                    'Iterations': 'mean'
                }).round(3)
                st.dataframe(summary_stats)
            else:
                st.info("Run more queries to see performance trends and analytics.")

if __name__ == "__main__":
    main()