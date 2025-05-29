1. What Data Are We Modeling?
Source: The data comes from a file like data/monthly_data.csv.
Structure: Each row represents a time period (e.g., a month or week).
Columns:
Date/Time column: e.g., month, week, or date
Sales column: e.g., sales (the target variable)
Marketing spend columns: e.g., facebook_spend, tv_spend, print_spend, ooh_spend, radio_spend, tiktok_spend, branded_search_spend, nonbranded_search_spend, etc.
Example:
| month | sales | facebook_spend | tv_spend | ... |
|---------|---------|----------------|----------|-----|
| 2023-01 | 100000 | 20000 | 15000 | ... |
| 2023-02 | 120000 | 25000 | 18000 | ... |
| ... | ... | ... | ... | ... |
Goal: To understand how each marketing channel’s spend affects sales, and to optimize future budget allocation.
2. What Modeling Technique Is Used?
a. Data Preparation
Adstock Transformation:
Each marketing spend channel is “adstocked” to account for carryover effects (i.e., the impact of marketing spend lasts for several periods, not just one).
This is a standard MMM technique.
b. Model Types
The system supports several regression model types:
Linear Regression:
Sales=β0+β1*channel1 +β2*channel2
loglog-regression
log(Sales)=β0+β log(Channel 1)+...+ϵ
semilogregression
Semi-Log Regression:
log(Sales)=β0+β1×Channel1+.+ϵlog(Sales)=β 0+β 1×Channel 1+...+ϵ

Features: The adstocked marketing spend columns (e.g., facebook_spend_adstocked, etc.)
d marketing spend columns (e.g., facebook_spend_adstocked, etc.)
Target: sales
c. Model Fitting
The code uses Ordinary Least Squares (OLS) regression from the statsmodels library.
It fits the model on a training set and evaluates on a test set.
Performance metrics: R², RMSE, MAPE, AIC, BIC, VIF (for multicollinearity).
d. Model Selection
The best model is chosen based on the highest R² on the test set.
e. Channel Contribution & ROI
The model’s coefficients are used to estimate each channel’s contribution to sales and its ROI.
f. Budget Optimization
The system uses the model’s ROI estimates to recommend how to allocate a given marketing budget across channels for maximum impact.
1. What is the percentage being shown on the pie chart?
The percentage shown for each channel is:
Percentage
=
Allocated Budget for Channel
Total Budget
×
100
Percentage= 
Total Budget
Allocated Budget for Channel
​
 ×100
For example, if facebook_spend is $825,685.53 and your total budget is $1,067,475.77, then:
Percentage for Facebook
=
825
,
685.53
1
,
067
,
475.77
×
100
≈
77.3
%
Percentage for Facebook= 
1,067,475.77
825,685.53
​
 ×100≈77.3%
This tells you what share of your total budget is being allocated to each channel.