
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Marketing Mix Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
''', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Marketing Mix Modeling Dashboard</h1>', unsafe_allow_html=True)

# Introduction
st.markdown('''
This dashboard presents the results of a Marketing Mix Modeling (MMM) analysis, showing how different marketing 
channels contribute to sales and providing recommendations for optimal budget allocation.
''')

# Dashboard sections
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Analysis</h2>', unsafe_allow_html=True)
    st.markdown('''
    The Analysis section provides insights from exploratory data analysis:
    
    - Time series trends of sales and marketing spend
    - Correlation between marketing channels and sales
    - Distribution of marketing spend
    - Adstock transformation results
    ''')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Key Metrics</h2>', unsafe_allow_html=True)
    
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.markdown('<p class="metric-label">Total Channels</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">8</p>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<p class="metric-label">Data Points</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">61</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Modeling</h2>', unsafe_allow_html=True)
    st.markdown('''
    The Modeling section shows the results of marketing mix models:
    
    - Model performance metrics
    - Channel contributions to sales
    - Return on investment (ROI) by channel
    - Comparison of different model types
    ''')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Optimization</h2>', unsafe_allow_html=True)
    st.markdown('''
    The Optimization section provides budget allocation recommendations:
    
    - Optimal allocation of marketing budget
    - Expected sales lift from optimized allocation
    - Interactive budget simulation tool
    - Comparison with current allocation
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

# How to use
st.markdown('<h2 class="sub-header">How to Use This Dashboard</h2>', unsafe_allow_html=True)
st.markdown('''
1. **Explore the Analysis page** to understand sales trends and channel performance
2. **Review the Modeling page** to see which channels contribute most to sales
3. **Use the Optimization page** to determine the optimal budget allocation
4. **Simulate different scenarios** by adjusting the total budget
''')

# Footer
st.markdown('---')
st.markdown('Marketing Mix Modeling Dashboard | Created with Streamlit')
