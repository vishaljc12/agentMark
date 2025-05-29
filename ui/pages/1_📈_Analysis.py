
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Analysis - Marketing Mix Modeling",
    page_icon="📈",
    layout="wide",
)

# Apply custom CSS
st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .insight-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin: 1rem 0;
    }
</style>
''', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Marketing Data Analysis</h1>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load original data
        data = pd.read_csv("data/monthly_data.csv")
        
        # Try to load analysis results
        try:
            with open("outputs/eda/analysis_results.json", "r") as f:
                analysis_results = json.load(f)
        except:
            analysis_results = {}
        
        return data, analysis_results
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}

data, analysis_results = load_data()

if data.empty:
    st.warning("No data loaded. Please make sure the data file exists and is valid.")
    st.stop()

# Display data overview
st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Marketing Channels")
    date_cols = [col for col in data.columns if col.lower() in ['week', 'date', 'month', 'time']]
    sales_cols = [col for col in data.columns if col.lower() in ['sales', 'revenue']]
    
    if date_cols and sales_cols:
        date_col = date_cols[0]
        sales_col = sales_cols[0]
        channels = [col for col in data.columns if col not in date_cols and col not in sales_cols]
        
        st.write(f"Number of marketing channels: {len(channels)}")
        st.write("Channels:", ", ".join(channels))
    else:
        st.warning("Could not identify date or sales columns.")

with col2:
    st.markdown("### Time Period")
    if date_cols:
        date_col = date_cols[0]
        st.write(f"Number of time periods: {len(data)}")
        st.write(f"First period: {data[date_col].iloc[0]}")
        st.write(f"Last period: {data[date_col].iloc[-1]}")
    else:
        st.warning("Could not identify date column.")

# Summary statistics
st.markdown('<h2 class="sub-header">Summary Statistics</h2>', unsafe_allow_html=True)
st.dataframe(data.describe())

# Time series visualizations
st.markdown('<h2 class="sub-header">Sales Trends</h2>', unsafe_allow_html=True)
if os.path.exists("ui/assets/sales_time_series.png"):
    st.image("ui/assets/sales_time_series.png", use_column_width=True)
else:
    st.warning("Sales trend visualization not found.")

# Channel spend over time
st.markdown('<h2 class="sub-header">Marketing Spend by Channel</h2>', unsafe_allow_html=True)
if os.path.exists("ui/assets/channel_spend_time_series.png"):
    st.image("ui/assets/channel_spend_time_series.png", use_column_width=True)
else:
    st.warning("Channel spend visualization not found.")

# Total spend by channel
st.markdown('<h2 class="sub-header">Total Spend Distribution</h2>', unsafe_allow_html=True)
if os.path.exists("ui/assets/total_spend_by_channel.png"):
    st.image("ui/assets/total_spend_by_channel.png", use_column_width=True)
else:
    st.warning("Total spend distribution visualization not found.")

# Correlation analysis
st.markdown('<h2 class="sub-header">Correlation Analysis</h2>', unsafe_allow_html=True)
if os.path.exists("ui/assets/correlation_heatmap.png"):
    st.image("ui/assets/correlation_heatmap.png", use_column_width=True)
else:
    st.warning("Correlation heatmap not found.")

# Display correlations with sales
st.markdown("### Correlation with Sales")
if "correlations" in analysis_results.get("eda", {}):
    correlations = analysis_results["eda"]["correlations"]
    corr_df = pd.DataFrame({
        "Channel": list(correlations.keys()),
        "Correlation with Sales": list(correlations.values())
    })
    # Filter out sales correlation with itself
    if sales_cols:
        sales_col = sales_cols[0]
        corr_df = corr_df[corr_df["Channel"] != sales_col]
    
    corr_df = corr_df.sort_values(
        by="Correlation with Sales", ascending=False
    )
    st.dataframe(corr_df)
else:
    # Calculate correlations directly if not in analysis results
    if sales_cols:
        sales_col = sales_cols[0]
        numeric_data = data.select_dtypes(include=["number"])
        if sales_col not in numeric_data.columns:
            st.warning("Sales column not found in numeric columns. Cannot calculate correlations.")
        else:
            corr = numeric_data.corr()[sales_col].drop(sales_col)
            corr_df = pd.DataFrame({
                "Channel": corr.index,
                "Correlation with Sales": corr.values
            }).sort_values(by="Correlation with Sales", ascending=False)
            st.dataframe(corr_df)
    else:
        st.warning("Sales column not identified. Cannot calculate correlations.")

# Sales vs channels
st.markdown('<h2 class="sub-header">Sales vs Marketing Channels</h2>', unsafe_allow_html=True)
if os.path.exists("ui/assets/sales_vs_channels.png"):
    st.image("ui/assets/sales_vs_channels.png", use_column_width=True)
else:
    st.warning("Sales vs channels visualization not found.")

# Adstock transformation
st.markdown('<h2 class="sub-header">Adstock Transformation</h2>', unsafe_allow_html=True)

# Load adstock parameters
@st.cache_data
def load_adstock_params():
    try:
        with open("outputs/eda/adstock_parameters.json", "r") as f:
            adstock_params = json.load(f)
        return adstock_params
    except:
        return {}

adstock_params = load_adstock_params()

if adstock_params:
    # Display adstock parameters
    adstock_df = pd.DataFrame({
        "Channel": list(adstock_params.keys()),
        "Optimal Decay Rate": [params["decay_rate"] for params in adstock_params.values()],
        "Correlation after Adstock": [params["correlation"] for params in adstock_params.values()]
    })
    adstock_df = adstock_df.sort_values(by="Correlation after Adstock", ascending=False)
    st.dataframe(adstock_df)
else:
    st.warning("Adstock parameters not found.")

# Key insights
st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)

# Generate basic insights if no custom ones are provided
insights = []

# Add top channels by correlation if available
if not insights and "correlations" in analysis_results.get("eda", {}) and sales_cols:
    sales_col = sales_cols[0]
    correlations = {k: v for k, v in analysis_results["eda"]["correlations"].items() if k != sales_col}
    
    if correlations:
        top_channels = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_channels_text = ", ".join([f"{channel} ({corr:.2f})" for channel, corr in top_channels])
        
        insights.append({
            "title": "Top Correlated Channels",
            "content": f"Based on correlation analysis, the channels most correlated with sales are: {top_channels_text}"
        })

# Add insight about adstock effects if available
if not insights and adstock_params:
    top_adstock = sorted(adstock_params.items(), key=lambda x: x[1]["decay_rate"], reverse=True)[:3]
    top_adstock_text = ", ".join([f"{channel} ({params['decay_rate']:.2f})" for channel, params in top_adstock])
    
    insights.append({
        "title": "Adstock Effects",
        "content": f"Channels with highest carryover effects (adstock decay rates) are: {top_adstock_text}, indicating their impact lasts longer than other channels."
    })

# Add a generic insight about seasonal patterns
if not insights:
    insights.append({
        "title": "Seasonal Patterns",
        "content": "The data may show seasonal patterns in sales. Analyzing these patterns can help optimize campaign timing."
    })

# Add a generic insight about spending distribution
if not insights:
    insights.append({
        "title": "Spend Distribution",
        "content": "There's variation in spend allocation across channels, with some potentially being under or over-utilized based on their correlation with sales."
    })

for i, insight in enumerate(insights):
    st.markdown(f'''
    <div class="insight-card">
        <h3>{insight["title"]}</h3>
        <p>{insight["content"]}</p>
    </div>
    ''', unsafe_allow_html=True)

# Footer
st.markdown('---')
st.markdown('Marketing Mix Modeling Dashboard | Analysis Page')
