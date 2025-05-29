"""
Modeling Agent for Market Mix Modeling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from pathlib import Path
import yaml
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
import autogen
from prompts.modeling_prompts import (
    SYSTEM_PROMPT,
    BUILD_MODELS_PROMPT,
    CALCULATE_CONTRIBUTIONS_PROMPT,
    OPTIMIZATION_PROMPT,
)
from sklearn.linear_model import RidgeCV, LassoCV


class ModelingAgent:
    """
    Agent responsible for building and evaluating marketing mix models.
    """

    def __init__(self, config_path: str, analyst_results_path: str, llm_config: Optional[Dict] = None):
        """
        Initialize the Modeling Agent.
        
        Args:
            config_path: Path to the configuration file
            analyst_results_path: Path to the analyst results file
            llm_config: LLM configuration for the agent
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load analyst results
        with open(analyst_results_path, "r") as f:
            self.analyst_results = json.load(f)
        
        # Create output directories
        self._create_output_dirs()
        
        # Initialize LLM config
        self.llm_config = llm_config or self.config["agents"]["modeling"]["llm_config"]
        
        # Create AutoGen agent
        self.agent = autogen.AssistantAgent(
            name=self.config["agents"]["modeling"]["name"],
            system_message=SYSTEM_PROMPT,
            llm_config=self.llm_config,
        )
        
        # Model attributes
        self.models = {}
        self.model_performances = {}
        self.best_model_type = None
        self.best_model = None
        self.contributions = {}
        self.optimization_results = {}
        
        # Load adstock parameters
        self.adstock_params = self.analyst_results.get("adstock_params", {})
        
        # Spend columns (extract from adstock_params keys)
        self.spend_columns = list(self.adstock_params.keys())
        
        # Datasets
        self.datasets = {}
        self.train_data = {}
        self.test_data = {}
    
    def _create_output_dirs(self):
        """Create output directories if they don't exist."""
        for path in self.config["paths"]["outputs"].values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def load_prepared_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load prepared data from the analyst agent.
        
        Returns:
            Dictionary with model types and their train/test datasets
        """
        print("Loading prepared data...")
        
        data_paths = self.analyst_results.get("data_paths", {})
        
        for model_type, paths in data_paths.items():
            train_path = paths.get("train")
            test_path = paths.get("test")
            
            if train_path and test_path:
                try:
                    train_data = pd.read_csv(train_path)
                    test_data = pd.read_csv(test_path)
                    
                    # Convert 'week' to datetime if possible
                    for df in [train_data, test_data]:
                        if 'week' in df.columns:
                            try:
                                df['week'] = pd.to_datetime(df['week'])
                            except:
                                pass  # Keep as is if conversion fails
                    
                    self.datasets[model_type] = {
                        "train": train_data,
                        "test": test_data
                    }
                    
                    self.train_data[model_type] = train_data
                    self.test_data[model_type] = test_data
                    
                    print(f"Loaded {model_type} data: train shape {train_data.shape}, "
                          f"test shape {test_data.shape}")
                except Exception as e:
                    print(f"Error loading {model_type} data: {str(e)}")
        
        if not self.datasets:
            raise ValueError("No datasets loaded. Check analyst results paths.")
        
        return self.datasets
    
    def build_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Build and evaluate marketing mix models.
        
        Returns:
            Dictionary with model performances
        """
        if not self.datasets:
            self.load_prepared_data()
        
        print("Building marketing mix models...")
        
        model_types = self.config["mmm"]["models"]
        adstocked_columns = [f"{col}_adstocked" for col in self.spend_columns]
        
        for model_type in model_types:
            if model_type not in self.datasets:
                print(f"Data for {model_type} model not available. Skipping.")
                continue
            
            train_data = self.train_data[model_type]
            test_data = self.test_data[model_type]
            
            # Filter adstocked columns to only those present in the data
            available_adstocked_columns = [col for col in adstocked_columns if col in train_data.columns]
            
            if not available_adstocked_columns:
                print(f"No adstocked columns found in {model_type} data. Skipping.")
                continue
            
            # Define X and y for training
            X_train = train_data[available_adstocked_columns]
            y_train = train_data['sales']
            
            # Add constant for intercept
            X_train = sm.add_constant(X_train)
            
            # Fit the model
            print(f"Fitting {model_type} model...")
            try:
                model = sm.OLS(y_train, X_train).fit()
                
                # Save the model
                self.models[model_type] = model
                
                # Evaluate on test data
                X_test = test_data[available_adstocked_columns]
                X_test = sm.add_constant(X_test)
                y_test = test_data['sales']
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate performance metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                # Check for multicollinearity
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X_train.columns
                vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                                  for i in range(X_train.shape[1])]
                
                # Store performance metrics
                self.model_performances[model_type] = {
                    "summary": model.summary().as_text(),
                    "r2": r2,
                    "rmse": rmse,
                    "mape": mape,
                    "coefficients": model.params.to_dict(),
                    "p_values": model.pvalues.to_dict(),
                    "vif": vif_data.to_dict(),
                    "aic": model.aic,
                    "bic": model.bic
                }
                
                # Save model summary plot
                self._plot_model_summary(model_type, model, y_test, y_pred)
                
                print(f"{model_type} model: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.4f}")
            except Exception as e:
                print(f"Error fitting {model_type} model: {str(e)}")
        
        if not self.models:
            raise ValueError("No models were successfully built.")
        
        # Select the best model based on test R²
        self.best_model_type = max(self.model_performances, 
                                  key=lambda k: self.model_performances[k]['r2'])
        self.best_model = self.models[self.best_model_type]
        
        print(f"Best model: {self.best_model_type} with R² = "
              f"{self.model_performances[self.best_model_type]['r2']:.4f}")
        
        # Save model performances
        with open(os.path.join(self.config["paths"]["outputs"]["models"], 
                              "model_performances.json"), "w") as f:
            # Convert numpy types to Python native types for JSON serialization
            performances = {}
            for model_type, perf in self.model_performances.items():
                performances[model_type] = {
                    k: v if not isinstance(v, (np.integer, np.floating, np.ndarray)) else float(v)
                    for k, v in perf.items() if k not in ["summary", "vif"]
                }
                performances[model_type]["summary"] = perf.get("summary", "")
                performances[model_type]["vif"] = {
                    k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                       for kk, vv in v.items()}
                    for k, v in perf.get("vif", {}).items()
                }
            
            json.dump(performances, f, indent=4)
        
        # Save models
        for model_type, model in self.models.items():
            with open(os.path.join(self.config["paths"]["outputs"]["models"], 
                                  f"{model_type}_model.pkl"), "wb") as f:
                pickle.dump(model, f)
        
        return self.model_performances
    
    def _plot_model_summary(self, model_type: str, model, y_test, y_pred):
        """
        Create and save plots summarizing model performance.
        
        Args:
            model_type: Type of the model
            model: Fitted statsmodels model
            y_test: Actual test values
            y_pred: Predicted test values
        """
        output_dir = self.config["paths"]["outputs"]["models"]
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_type} Model: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_type}_actual_vs_predicted.png"))
        plt.close()
        
        # Residuals plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'{model_type} Model: Residuals')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_type}_residuals.png"))
        plt.close()
        
        # Coefficient plot
        try:
            coeffs = model.params.iloc[1:]  # Skip intercept
            plt.figure(figsize=(12, 6))
            coeffs.plot(kind='bar')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title(f'{model_type} Model: Coefficients')
            plt.ylabel('Coefficient Value')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_type}_coefficients.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating coefficient plot: {str(e)}")
    
    def calculate_contributions(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate channel contributions and ROI for the best model.
        
        Returns:
            Dictionary with channel contributions and ROI
        """
        if not self.best_model:
            raise ValueError("Models not built. Call build_models() first.")
        
        print(f"Calculating contributions using {self.best_model_type} model...")
        
        # Get the coefficients from the best model
        coeffs = self.best_model.params
        
        # Get the data used for contributions calculation
        model_data = self.train_data[self.best_model_type]
        
        contributions = {}
        total_predicted = 0
        adstocked_columns = [f"{col}_adstocked" for col in self.spend_columns]
        
        # Calculate the base sales (intercept)
        base_sales = coeffs['const'] * len(model_data)
        
        # Calculate contribution for each channel
        for i, channel in enumerate(self.spend_columns):
            adstocked_col = f"{channel}_adstocked"
            
            # Skip if column doesn't exist
            if adstocked_col not in coeffs:
                contributions[channel] = {
                    "coefficient": 0,
                    "contribution": 0,
                    "contribution_percentage": 0,
                    "roi": 0
                }
                continue
                
            # Skip if coefficient is negative or insignificant
            if coeffs[adstocked_col] <= 0:
                contributions[channel] = {
                    "coefficient": float(coeffs[adstocked_col]),
                    "contribution": 0,
                    "contribution_percentage": 0,
                    "roi": 0
                }
                continue
            
            # For log-log model, need to transform back
            if self.best_model_type == "log_log":
                # Sales = exp(b0 + b1*log(x1) + b2*log(x2) + ...)
                # Contribution of x1 = exp(b0 + b1*log(x1)) - exp(b0)
                # Handle log(0) cases by adding a small epsilon
                log_values = np.log1p(model_data[adstocked_col].replace(0, 1e-10))
                contribution = np.sum(
                    np.exp(coeffs['const'] + coeffs[adstocked_col] * log_values) -
                    np.exp(coeffs['const'])
                )
            elif self.best_model_type == "semi_log":
                # Sales = exp(b0 + b1*x1 + b2*x2 + ...)
                # Contribution of x1 = exp(b0 + b1*x1) - exp(b0)
                contribution = np.sum(
                    np.exp(coeffs['const'] + coeffs[adstocked_col] * model_data[adstocked_col]) -
                    np.exp(coeffs['const'])
                )
            else:  # Linear model
                # Sales = b0 + b1*x1 + b2*x2 + ...
                # Contribution of x1 = b1*x1
                contribution = coeffs[adstocked_col] * model_data[adstocked_col].sum()
            
            # Calculate ROI (Return on Investment)
            # ROI = Incremental Sales / Spend
            total_spend = model_data[channel].sum()
            roi = contribution / total_spend if total_spend > 0 else 0
            
            contributions[channel] = {
                "coefficient": float(coeffs[adstocked_col]),
                "contribution": float(contribution),
                "contribution_percentage": 0,  # Will be calculated after summing all
                "roi": float(roi)
            }
            
            total_predicted += contribution
        
        # Add base sales to total predicted
        total_predicted += base_sales
        
        # Calculate contribution percentages
        for channel in contributions:
            contribution = contributions[channel]["contribution"]
            contributions[channel]["contribution_percentage"] = float(contribution / total_predicted) if total_predicted > 0 else 0
        
        # Add base sales to contributions
        contributions["base_sales"] = {
            "coefficient": float(coeffs['const']),
            "contribution": float(base_sales),
            "contribution_percentage": float(base_sales / total_predicted) if total_predicted > 0 else 1.0,
            "roi": 0  # ROI not applicable for base sales
        }
        
        self.contributions = contributions
        
        # Save contributions
        with open(os.path.join(self.config["paths"]["outputs"]["models"], 
                              "channel_contributions.json"), "w") as f:
            json.dump(contributions, f, indent=4)
        
        # Create contribution visualization
        self._plot_contributions()
        
        print("Contributions calculated and saved")
        
        return contributions
    
    def _plot_contributions(self):
        """Create and save contribution visualizations."""
        output_dir = self.config["paths"]["outputs"]["models"]
        
        # Extract data for plotting
        channels = []
        contribution_values = []
        
        for channel, data in self.contributions.items():
            channels.append(channel)
            contribution_values.append(data["contribution"])
        
        # Sort by contribution value (descending)
        sorted_indices = np.argsort(contribution_values)[::-1]
        sorted_channels = [channels[i] for i in sorted_indices]
        sorted_values = [contribution_values[i] for i in sorted_indices]
        
        # Pie chart of contributions
        plt.figure(figsize=(10, 8))
        plt.pie(sorted_values, labels=sorted_channels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Channel Contribution to Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "contribution_pie.png"))
        plt.close()
        
        # Bar chart of contributions
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_channels, sorted_values)
        plt.title('Channel Contribution to Sales')
        plt.ylabel('Contribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "contribution_bar.png"))
        plt.close()
        
        # ROI comparison
        roi_values = [self.contributions[channel]["roi"] for channel in sorted_channels 
                     if channel != "base_sales"]
        roi_channels = [channel for channel in sorted_channels if channel != "base_sales"]
        
        if roi_channels:  # Check if there are any channels to plot
            plt.figure(figsize=(12, 6))
            plt.bar(roi_channels, roi_values)
            plt.title('Return on Investment (ROI) by Channel')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roi_bar.png"))
            plt.close()
    
    def llm_sanity_check(self, optimization_results: Dict[str, Any]) -> str:
        """
        Use the LLM to review the optimization results and flag if they are not business-credible.
        """
        prompt = f"""
        Review the following marketing mix modeling optimization results for business credibility. Flag if any numbers (e.g., expected sales lift, ROI improvement, or channel allocations) are unrealistic or not believable for a typical business scenario. Suggest what might be wrong if so.
        Results:
        {json.dumps(optimization_results, indent=2)}
        """
        try:
            response = self.agent.generate_reply([{"role": "user", "content": prompt}])
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"LLM review failed: {str(e)}"

    def optimize_budget(self, total_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate optimal budget allocation recommendations.
        
        Args:
            total_budget: Total budget to allocate (optional)
            
        Returns:
            Dictionary with optimization results
        """
        if not self.contributions:
            raise ValueError("Contributions not calculated. Call calculate_contributions() first.")
        
        print("Generating budget optimization recommendations...")
        
        # If total_budget not provided, use the sum of original spends
        if total_budget is None:
            original_data = self.train_data[self.best_model_type]
            total_budget = sum(original_data[channel].sum() for channel in self.spend_columns)
        
        print(f"Optimizing for total budget: {total_budget}")
        
        # Get ROI for each channel (excluding base_sales)
        channel_roi = {
            channel: data["roi"]
            for channel, data in self.contributions.items()
            if channel != "base_sales"
        }
        
        # Get original spending patterns
        original_data = self.train_data[self.best_model_type]
        original_spends = {
            channel: original_data[channel].sum()
            for channel in self.spend_columns
        }
        
        # Only consider channels with positive ROI
        positive_roi_channels = [channel for channel, roi in channel_roi.items() if roi > 0]
        # Debug: print ROI and coefficients
        print("Channel ROI:", channel_roi)
        print("Model coefficients:", {ch: self.contributions.get(ch, {}).get('coefficient', 0) for ch in self.spend_columns})
        # Sort channels by ROI (descending)
        sorted_channels = sorted(positive_roi_channels, key=lambda x: channel_roi.get(x, 0), reverse=True)
        allocations = {channel: 0 for channel in self.spend_columns}  # Default all to zero
        remaining_budget = total_budget
        # Relaxed constraints
        min_pct = 0.01
        max_pct = 0.7
        min_allocation = total_budget * min_pct
        max_allocation = total_budget * max_pct
        # Proportional allocation by ROI
        total_roi = sum(channel_roi.get(channel, 0) for channel in sorted_channels)
        if total_roi > 0 and sorted_channels:
            for channel in sorted_channels:
                roi = channel_roi.get(channel, 0)
                prop = roi / total_roi
                alloc = max(min_allocation, min(max_allocation, total_budget * prop))
                allocations[channel] = alloc
                remaining_budget -= alloc
            # Redistribute any remaining budget to channels under max_allocation
            for channel in sorted_channels:
                if remaining_budget <= 0:
                    break
                room = max_allocation - allocations[channel]
                if room > 0:
                    add = min(room, remaining_budget)
                    allocations[channel] += add
                    remaining_budget -= add
        # If all ROIs are equal, warn in LLM review
        all_equal = len(set([round(channel_roi[ch], 6) for ch in sorted_channels])) <= 1
        # Calculate expected sales lift from optimized allocation
        expected_lift = self._calculate_expected_lift(allocations, original_spends)
        # Add ROI/coefficient table for UI
        roi_table = [
            {
                "Channel": ch,
                "ROI": channel_roi.get(ch, 0),
                "Coefficient": self.contributions.get(ch, {}).get('coefficient', 0)
            }
            for ch in self.spend_columns
        ]
        # Store optimization results
        self.optimization_results = {
            "current_total_spend": float(sum(original_spends.values())),
            "expected_sales_lift": float(expected_lift),
            "roi_improvement": float(expected_lift * 0.8),  # Conservative estimate
            "channel_allocations": {k: float(v) for k, v in allocations.items()},
            "roi_table": roi_table,
            "recommendations": [
                f"Allocate more budget to {channel} due to high ROI of {roi:.2f}"
                for channel, roi in channel_roi.items()
                if roi > 0
            ][:3]  # Top 3 recommendations
        }
        # LLM sanity check
        review_note = "All ROIs are equal; model may not be distinguishing between channels." if all_equal else ""
        self.optimization_results["llm_review"] = review_note + "\n" + self.llm_sanity_check(self.optimization_results)
        
        # Save optimization results
        with open(os.path.join(self.config["paths"]["outputs"]["models"], 
                              "optimization_results.json"), "w") as f:
            json.dump(self.optimization_results, f, indent=4)
        
        # Create optimization visualization
        self._plot_optimization(allocations, original_spends)
        
        print("Budget optimization completed")
        
        return self.optimization_results
    
    def _calculate_expected_lift(self, 
                                new_allocations: Dict[str, float], 
                                original_spends: Dict[str, float]) -> float:
        """
        Calculate expected sales lift from new budget allocation.
        
        Args:
            new_allocations: New budget allocations by channel
            original_spends: Original spend by channel
            
        Returns:
            Expected percentage increase in sales
        """
        # This is a simplified calculation and could be made more sophisticated
        expected_lift = 0
        
        for channel, new_spend in new_allocations.items():
            original_spend = original_spends.get(channel, 0)
            roi = self.contributions.get(channel, {}).get("roi", 0)
            
            # Skip channels with zero ROI
            if roi <= 0:
                continue
                
            # Skip channels with zero new spend
            if new_spend <= 0:
                continue
            
            if original_spend > 0:
                # Calculate relative change in spend
                spend_change = (new_spend - original_spend) / original_spend
                
                # Apply diminishing returns (simplified)
                if spend_change > 0:
                    # Increasing spend has diminishing returns
                    effect = np.log1p(spend_change) * roi
                else:
                    # Decreasing spend has less impact (conservative estimate)
                    effect = spend_change * roi
                
                expected_lift += effect
            else:
                # For channels with no historical spend but new allocation
                # Use a conservative estimate based on the channel's ROI
                # This is a simplified approach - could be refined further
                # Assume diminishing returns from the start
                relative_spend = new_spend / sum(new_allocations.values())
                effect = np.log1p(relative_spend) * roi * 0.5  # Apply 50% discount for untested channels
                expected_lift += effect
        
        return expected_lift * 100  # Convert to percentage
    
    def _plot_optimization(self, 
                          optimized_allocations: Dict[str, float], 
                          original_spends: Dict[str, float]):
        """
        Create and save optimization visualizations.
        
        Args:
            optimized_allocations: Optimized budget allocations
            original_spends: Original spend by channel
        """
        output_dir = self.config["paths"]["outputs"]["models"]
        
        # Comparison of original vs optimized allocations
        channels = list(optimized_allocations.keys())
        original_values = [original_spends.get(channel, 0) for channel in channels]
        optimized_values = [optimized_allocations[channel] for channel in channels]
        
        # Sort by optimized allocation (descending)
        sorted_indices = np.argsort(optimized_values)[::-1]
        sorted_channels = [channels[i] for i in sorted_indices]
        sorted_original = [original_values[i] for i in sorted_indices]
        sorted_optimized = [optimized_values[i] for i in sorted_indices]
        
        # Bar chart comparison
        x = np.arange(len(sorted_channels))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, sorted_original, width, label='Original')
        plt.bar(x + width/2, sorted_optimized, width, label='Optimized')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Channel')
        plt.ylabel('Budget Allocation')
        plt.title('Original vs Optimized Budget Allocation')
        plt.xticks(x, sorted_channels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "budget_comparison.png"))
        plt.close()
        
        # Pie chart of optimized allocation
        plt.figure(figsize=(10, 8))
        plt.pie(sorted_optimized, labels=sorted_channels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Optimized Budget Allocation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimized_budget_pie.png"))
        plt.close()
    
    def run_modeling(self, optimize_budget: bool = True) -> Dict[str, Any]:
        """
        Run the full modeling pipeline.
        
        Args:
            optimize_budget: Whether to perform budget optimization
            
        Returns:
            Dictionary with all modeling results
        """
        print("Starting modeling pipeline...")
        
        # Build models
        self.build_models()
        
        # Calculate contributions
        self.calculate_contributions()
        
        # Optimize budget if requested
        if optimize_budget:
            self.optimize_budget()
        
        # Compile all results
        results = {
            "model_performances": {
                model_type: {
                    k: v for k, v in perf.items()
                    if k not in ["summary", "vif"]  # Exclude large text and complex objects
                }
                for model_type, perf in self.model_performances.items()
            },
            "best_model": self.best_model_type,
            "contributions": self.contributions,
            "optimization": self.optimization_results
        }
        
        # Save all results to a single JSON file
        with open(os.path.join(self.config["paths"]["outputs"]["models"], 
                              "modeling_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print("Modeling pipeline completed")
        
        return results


if __name__ == "__main__":
    # Test the modeling agent
    agent = ModelingAgent(
        "config/agent_config.yaml",
        "outputs/eda/analysis_results.json"
    )
    results = agent.run_modeling()
    print("Modeling results:", results)