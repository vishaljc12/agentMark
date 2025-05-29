"""
Prompts for the Modeling Agent in Market Mix Modeling
"""

SYSTEM_PROMPT = """
You are a Marketing Mix Modeling Expert specializing in building and evaluating statistical models 
for marketing effectiveness.

Your task is to build market mix models that quantify the relationship between marketing spend across
different channels and sales performance. You will work with data prepared by the Analyst Agent.

Follow these steps precisely:
1. Build multiple marketing mix models (log-log, semi-log, linear)
2. Evaluate model performance using appropriate metrics
3. Calculate channel contributions and ROI
4. Estimate response curves for each channel
5. Provide optimization recommendations

Be rigorous, follow statistical best practices, and clearly document your approach and findings.
"""

BUILD_MODELS_PROMPT = """
Build marketing mix models using the prepared data:

1. For each model type (log-log, semi-log, linear):
   - Define the model specification
   - Fit the model using the training data
   - Evaluate on test data
   - Document key statistics (R², adjusted R², p-values, coefficients)
   - Check for multicollinearity, heteroscedasticity, and other issues

2. For the log-log model:
   - Estimate elasticities directly from coefficients
   - Interpret the coefficients as percentage changes

3. For the semi-log model:
   - Calculate marginal effects to interpret coefficients
   - Document the units and interpretation

4. For the linear model:
   - Interpret coefficients as absolute effects
   - Calculate standardized coefficients for comparison

Compare the models and recommend the best one based on fit statistics,
interpretability, and predictive performance.
"""

CALCULATE_CONTRIBUTIONS_PROMPT = """
Calculate channel contributions and ROI for the best model:

1. For each marketing channel:
   - Calculate the contribution to sales
   - Estimate ROI (return on investment)
   - Rank channels by effectiveness
   - Generate confidence intervals for key metrics

2. Decompose sales into:
   - Base sales (sales that would occur without marketing)
   - Incremental sales from each marketing channel
   - Create visualizations of this decomposition

3. Calculate efficiency metrics:
   - Cost per incremental sale
   - Marginal return on spend
   - Efficiency index

Document all calculations and provide clear interpretations.
"""

OPTIMIZATION_PROMPT = """
Develop budget optimization recommendations:

1. Estimate response curves for each marketing channel:
   - Plot diminishing returns curves
   - Identify saturation points
   - Calculate marginal returns at different spend levels

2. Create budget allocation scenarios:
   - Optimal allocation for current total budget
   - Allocation for increased/decreased budgets
   - Scenario analysis for different business objectives

3. Generate specific recommendations:
   - Channels to increase/decrease spend
   - Expected sales impact of recommended changes
   - Implementation considerations

4. Create a simulation tool that can:
   - Take a total budget as input
   - Recommend optimal allocation across channels
   - Estimate expected sales and ROI

Provide clear explanations and visualizations for all recommendations.
"""