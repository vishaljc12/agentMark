"""
Prompts for the UI Agent in Market Mix Modeling
"""

SYSTEM_PROMPT = """
You are a UI Development Expert specializing in creating dashboards for marketing analytics.

Your task is to create a Streamlit dashboard for Market Mix Modeling results that will:
1. Display analysis insights and visualizations from the Analyst Agent
2. Show modeling results, contributions, and ROI from the Modeling Agent
3. Provide an interactive budget optimization tool

Create a professional, intuitive, and visually appealing dashboard that makes complex analytics 
accessible and actionable for marketing decision-makers.
"""

CREATE_DASHBOARD_PROMPT = """
Create a Streamlit dashboard with the following specifications:

1. Multi-page structure:
   - Home page with project overview and key metrics
   - Analysis page showing EDA results and insights
   - Modeling page with model performance and channel contributions
   - Optimization page with budget allocation recommendations and simulation

2. For the Analysis page:
   - Display time series of sales and marketing channels
   - Show correlation heatmap and key relationships
   - Include insights about channel performance and patterns

3. For the Modeling page:
   - Compare performance of different model types
   - Visualize channel contributions and ROI
   - Display model coefficients and interpretations

4. For the Optimization page:
   - Create an interactive budget allocation tool
   - Allow users to input total budget and simulate outcomes
   - Show recommended allocations vs. current allocations
   - Display expected sales lift and ROI

5. General requirements:
   - Professional design with consistent color scheme
   - Clear navigation between pages
   - Proper labeling and explanations for all visualizations
   - Mobile-responsive layout

Ensure the dashboard is user-friendly for marketing professionals who may not have technical expertise.
"""

OPTIMIZATION_TOOL_PROMPT = """
Create an interactive budget optimization tool with these features:

1. Input field for total marketing budget
2. Current allocation display (pie chart or bar chart)
3. Recommended allocation based on the optimization model
4. Expected impact on sales and ROI
5. Ability to manually adjust allocations and see impact
6. Comparison view of current vs. optimized allocation
7. Option to download the recommended allocation as CSV

The tool should be intuitive for marketing professionals to use without technical knowledge
of the underlying models. Include clear explanations of how the recommendations are generated
and what factors influence the optimization.
"""