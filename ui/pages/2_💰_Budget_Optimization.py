import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from main import run_budget_optimization

# Set page configuration
st.set_page_config(
    page_title="Budget Optimization - Marketing Mix Modeling",
    page_icon="💰",
    layout="centered",
)

# Custom CSS for a modern, professional look
st.markdown('''
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: #fff;
        border-radius: 1rem;
        box-shadow: 0 2px 16px rgba(30,136,229,0.08);
        padding: 2.5rem 2rem 2rem 2rem;
        margin: 2rem auto 2rem auto;
        max-width: 700px;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #616161;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .pie-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #424242;
        text-align: center;
        margin-top: 1.5rem;
    }
    .allocation-table {
        margin-top: 1.5rem;
    }
    .recommend-card {
        background: #f5f7fa;
        border-radius: 0.8rem;
        box-shadow: 0 1px 8px rgba(30,136,229,0.05);
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        margin: 2rem auto 2rem auto;
        max-width: 700px;
    }
</style>
''', unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">💰 Budget Optimizer</div>', unsafe_allow_html=True)

# --- Section 1: Historical Marketing Spend ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#1E88E5;'>Historical Marketing Spend</h2>", unsafe_allow_html=True)
    train_path = "outputs/eda/original_train.csv"
    if Path(train_path).exists():
        df_hist = pd.read_csv(train_path)
        channel_cols = [col for col in df_hist.columns if col.endswith('_spend') and not col.endswith('_adstocked')]
        hist_spend = df_hist[channel_cols].sum().to_dict()
        hist_total = sum(hist_spend.values())
        fig_hist = go.Figure(data=[go.Pie(
            labels=list(hist_spend.keys()),
            values=list(hist_spend.values()),
            hole=.35,
            marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='inside',
            textfont_size=13,
            insidetextorientation='horizontal',
        )])
        fig_hist.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            height=420,
            margin=dict(t=10, b=10, l=0, r=0)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('<div class="pie-title">Historical Budget Distribution</div>', unsafe_allow_html=True)
        hist_table = pd.DataFrame({
            "Channel": list(hist_spend.keys()),
            "Allocated Budget": [f"${v:,.2f}" for v in hist_spend.values()],
            "Percentage": [f"{(v/hist_total)*100:.1f}%" if hist_total > 0 else "0.0%" for v in hist_spend.values()]
        })
        st.markdown('<div class="allocation-table">', unsafe_allow_html=True)
        st.dataframe(hist_table, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Historical training data not found.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Section 2: Optimized Budget Allocation ---
# Dynamically get optimization results for the entered budget
@st.cache_data(show_spinner=False)
def get_optimization_results(budget):
    return run_budget_optimization(budget)

# Get current total spend from historical data as default
current_total = hist_total if 'hist_total' in locals() else 1000000
new_budget = st.number_input(
    "Enter new total marketing budget ($)",
    min_value=float(current_total * 0.5),
    max_value=float(current_total * 2.0),
    value=float(current_total),
    step=float(current_total * 0.05),
    format="%.2f",
    key="budget_input"
)
optimization_results = get_optimization_results(new_budget)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#1E88E5;'>Optimized Budget Allocation</h2>", unsafe_allow_html=True)
    if current_total > 0:
        budget_change = ((new_budget - current_total) / current_total) * 100
        st.markdown(f"<div style='text-align:center; color:#1E88E5; font-size:1.1rem;'>Budget change: <b>{budget_change:+.1f}%</b></div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid budget amount")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-label">Expected Sales Lift</div>', unsafe_allow_html=True)
        sales_lift = optimization_results.get("expected_sales_lift", 0)
        st.markdown(f'<div class="metric-value">{sales_lift:+.1f}%</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">ROI Improvement</div>', unsafe_allow_html=True)
        roi_improvement = optimization_results.get("roi_improvement", 0)
        st.markdown(f'<div class="metric-value">{roi_improvement:+.1f}%</div>', unsafe_allow_html=True)
    if "channel_allocations" in optimization_results:
        allocations = optimization_results["channel_allocations"]
        fig_opt = go.Figure(data=[go.Pie(
            labels=list(allocations.keys()),
            values=list(allocations.values()),
            hole=.35,
            marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='inside',
            textfont_size=13,
            insidetextorientation='horizontal',
        )])
        fig_opt.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            height=420,
            margin=dict(t=10, b=10, l=0, r=0),
            annotations=[dict(
                text='Optimized Budget Distribution',
                x=0.5, y=1.15, xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=22, color='#1E88E5', family='Arial', weight='bold'),
                align='center'
            )]
        )
        st.plotly_chart(fig_opt, use_container_width=True)
        opt_table = pd.DataFrame({
            "Channel": list(allocations.keys()),
            "Allocated Budget": [f"${v:,.2f}" for v in allocations.values()],
            "Percentage": [f"{(v/new_budget)*100:.1f}%" if new_budget > 0 else "0.0%" for v in allocations.values()]
        })
        st.markdown('<div class="allocation-table">', unsafe_allow_html=True)
        st.dataframe(opt_table, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Section 3: Comparison Table ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#1E88E5;'>Comparison: Historical vs. Optimized</h2>", unsafe_allow_html=True)
    if Path(train_path).exists() and "channel_allocations" in optimization_results:
        hist_dict = hist_spend
        opt_dict = optimization_results["channel_allocations"]
        all_channels = sorted(set(hist_dict.keys()) | set(opt_dict.keys()))
        comp_table = pd.DataFrame({
            "Channel": all_channels,
            "Historical Budget": [hist_dict.get(ch, 0.0) for ch in all_channels],
            "Optimized Budget": [opt_dict.get(ch, 0.0) for ch in all_channels]
        })
        comp_table["Historical %"] = comp_table["Historical Budget"] / hist_total * 100
        comp_table["Optimized %"] = comp_table["Optimized Budget"] / new_budget * 100
        comp_table["Change ($)"] = comp_table["Optimized Budget"] - comp_table["Historical Budget"]
        comp_table["Change (%)"] = comp_table["Optimized %"] - comp_table["Historical %"]
        comp_table["Historical Budget"] = comp_table["Historical Budget"].apply(lambda v: f"${v:,.2f}")
        comp_table["Optimized Budget"] = comp_table["Optimized Budget"].apply(lambda v: f"${v:,.2f}")
        comp_table["Historical %"] = comp_table["Historical %"].apply(lambda v: f"{v:.1f}%")
        comp_table["Optimized %"] = comp_table["Optimized %"].apply(lambda v: f"{v:.1f}%")
        comp_table["Change ($)"] = comp_table["Change ($)"].apply(lambda v: f"${v:,.2f}")
        comp_table["Change (%)"] = comp_table["Change (%)"].apply(lambda v: f"{v:.1f}%")
        st.dataframe(comp_table, hide_index=True, use_container_width=True)
    else:
        st.warning("Cannot display comparison: missing data.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Recommendations ---
with st.container():
    st.markdown('<div class="recommend-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#1E88E5;'>Recommendations</h2>", unsafe_allow_html=True)
    if "recommendations" in optimization_results:
        for rec in optimization_results["recommendations"]:
            st.markdown(f"- {rec}")
    else:
        st.markdown("""
- Consider reallocating budget to channels with higher ROI
- Monitor performance of channels with increased allocation
- Review channels with reduced budget for potential optimization
- Track the impact of budget changes on overall marketing effectiveness
""")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div style="text-align:center; color:#888; margin-top:2rem;">Marketing Mix Modeling Dashboard | Budget Optimization Page</div>', unsafe_allow_html=True) 