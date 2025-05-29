"""
Analyst Agent for Market Mix Modeling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple, Any, Optional
import autogen
from prompts.analyst_prompts import (
    SYSTEM_PROMPT,
    ANALYZE_DATA_PROMPT,
    ADSTOCK_TRANSFORMATION_PROMPT,
    DATA_PREPARATION_PROMPT,
)
from agents.utils import find_optimal_adstock_decay, apply_adstock, handle_missing_values, standardize_column_names


class AnalystAgent:
    """
    Agent responsible for data analysis, EDA, and preparing data for MMM.
    """

    def __init__(self, config_path: str, llm_config: Optional[Dict] = None):
        """
        Initialize the Analyst Agent.
        
        Args:
            config_path: Path to the configuration file
            llm_config: LLM configuration for the agent
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self._create_output_dirs()
        
        # Initialize LLM config
        self.llm_config = llm_config or self.config["agents"]["analyst"]["llm_config"]
        
        # Create AutoGen agent
        self.agent = autogen.AssistantAgent(
            name=self.config["agents"]["analyst"]["name"],
            system_message=SYSTEM_PROMPT,
            llm_config=self.llm_config,
        )
        
        # Data attributes
        self.data = None
        self.adstocked_data = None
        self.train_data = None
        self.test_data = None
        self.adstock_params = {}
        self.spend_columns = []
        self.results = {}
        
    def _create_output_dirs(self):
        """Create output directories if they don't exist."""
        for path in self.config["paths"]["outputs"].values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from sales and spend columns using the IQR method.
        """
        cols = ['sales'] + self.spend_columns
        for col in cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                after = len(df)
                if before != after:
                    print(f"Removed {before - after} outliers from {col}")
        return df
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            data_path: Path to the CSV file, defaults to config path
            
        Returns:
            Loaded DataFrame
        """
        data_path = data_path or self.config["paths"]["data"]
        print(f"Loading data from {data_path}")
        
        try:
            # Try to load with automatic detection of separator
            self.data = pd.read_csv(data_path, sep=None, engine='python')
        except Exception as e:
            print(f"Error with automatic separator detection: {str(e)}")
            # Fallback to standard CSV loading
            try:
                self.data = pd.read_csv(data_path)
            except Exception as e2:
                print(f"Error loading standard CSV: {str(e2)}")
                try:
                    # Try Excel format
                    self.data = pd.read_excel(data_path)
                except Exception as e3:
                    raise ValueError(f"Failed to load data: {str(e3)}")
        
        # Standardize column names
        self.data = standardize_column_names(self.data)
        
        # Check if 'week' column exists, otherwise try to use first column as date
        if 'week' not in self.data.columns and len(self.data.columns) > 1:
            # Rename first column to 'week'
            self.data = self.data.rename(columns={self.data.columns[0]: 'week'})
        
        # Check if 'sales' column exists (case-insensitive)
        sales_col = next((col for col in self.data.columns if col.lower() == 'sales'), None)
        if sales_col and sales_col != 'sales':
            # Rename to standardized 'sales'
            self.data = self.data.rename(columns={sales_col: 'sales'})
        
        # Identify spend columns (exclude 'week' and 'sales')
        self.spend_columns = [col for col in self.data.columns 
                             if col not in ['week', 'sales'] and not col.startswith('unnamed:')]
        
        # Convert 'week' to datetime if possible
        try:
            self.data['week'] = pd.to_datetime(self.data['week'])
        except:
            print("Warning: Unable to convert 'week' column to datetime. Treating as string.")
        
        # Handle missing or NaN values
        self.data = handle_missing_values(self.data)
        # Remove outliers
        self.data = self.remove_outliers(self.data)
        
        print(f"Data loaded with shape {self.data.shape}")
        print(f"Spend columns identified: {self.spend_columns}")
        
        return self.data
    
    def perform_eda(self) -> Dict[str, Any]:
        """
        Perform exploratory data analysis on the marketing data.
        
        Returns:
            Dictionary with EDA results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Performing exploratory data analysis...")
        
        # Create basic statistics
        stats = {
            "summary": self.data.describe().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "correlations": self.data.corr()['sales'].to_dict()
        }
        
        # Create visualizations
        self._create_eda_visualizations()
        
        # Store results
        self.results["eda"] = stats
        
        # Save results to JSON
        with open(os.path.join(self.config["paths"]["outputs"]["eda"], "eda_stats.json"), "w") as f:
            json.dump(stats, f, indent=4, default=str)
            
        print("EDA completed and saved")
        
        return stats
    
    def _create_eda_visualizations(self):
        """Create and save EDA visualizations."""
        output_dir = self.config["paths"]["outputs"]["eda"]
        
        # Time series plot of sales
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['week'], self.data['sales'])
        plt.title('Sales Over Time')
        plt.xlabel('Week')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sales_time_series.png"))
        plt.close()
        
        # Marketing spend by channel over time
        plt.figure(figsize=(15, 10))
        for i, channel in enumerate(self.spend_columns):
            plt.subplot(len(self.spend_columns)//2 + len(self.spend_columns)%2, 2, i+1)
            plt.plot(self.data['week'], self.data[channel])
            plt.title(f'{channel} Over Time')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "channel_spend_time_series.png"))
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        
        # Scatter plots of sales vs each channel
        plt.figure(figsize=(15, 10))
        for i, channel in enumerate(self.spend_columns):
            plt.subplot(len(self.spend_columns)//2 + len(self.spend_columns)%2, 2, i+1)
            plt.scatter(self.data[channel], self.data['sales'])
            plt.title(f'Sales vs {channel}')
            plt.xlabel(channel)
            plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sales_vs_channels.png"))
        plt.close()
        
        # Distribution of marketing spend by channel
        plt.figure(figsize=(12, 6))
        self.data[self.spend_columns].sum().plot(kind='bar')
        plt.title('Total Spend by Channel')
        plt.ylabel('Total Spend')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "total_spend_by_channel.png"))
        plt.close()
    
    def apply_adstock_transformation(self) -> pd.DataFrame:
        """
        Apply adstock transformation to marketing spend variables.
        
        Returns:
            DataFrame with adstocked variables
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Applying adstock transformation...")
        
        # Create a copy of the original data
        self.adstocked_data = self.data.copy()
        
        # Find optimal decay rates and apply transformations
        for channel in self.spend_columns:
            # Get decay range from config or use default
            decay_range = self.config["mmm"]["adstock_decay_ranges"].get(
                channel, self.config["mmm"].get("default_adstock_decay_range", [0.1, 0.9])
            )
            
            best_decay, best_corr = self._find_optimal_decay(channel, decay_range)
            self.adstock_params[channel] = {
                "decay_rate": best_decay,
                "correlation": best_corr
            }
            
            # Apply the optimal decay
            self.adstocked_data[f"{channel}_adstocked"] = self._apply_adstock(
                self.data[channel], best_decay
            )
        
        # Save adstock parameters
        with open(os.path.join(self.config["paths"]["outputs"]["eda"], 
                              "adstock_parameters.json"), "w") as f:
            json.dump(self.adstock_params, f, indent=4)
        
        print("Adstock transformation completed")
        
        return self.adstocked_data
    
    def _find_optimal_decay(self, channel: str, decay_range: List[float]) -> Tuple[float, float]:
        """
        Find optimal decay rate for a channel that maximizes correlation with sales.
        
        Args:
            channel: Channel name
            decay_range: Range of decay rates to try [min, max]
            
        Returns:
            Tuple of (best_decay_rate, best_correlation)
        """
        # Check if channel has any non-zero values
        if self.data[channel].sum() == 0 or self.data[channel].nunique() <= 1:
            print(f"Channel {channel} has no variation or only zero values. Using default decay rate.")
            return decay_range[0], 0
        
        decay_values = np.linspace(decay_range[0], decay_range[1], 20)
        best_decay = decay_range[0]
        best_corr = 0
        
        for decay in decay_values:
            adstocked = self._apply_adstock(self.data[channel], decay)
            
            # Handle case where adstocked values have no variation
            if np.std(adstocked) == 0:
                continue
                
            corr = abs(np.corrcoef(adstocked, self.data['sales'])[0, 1])
            
            # Handle NaN correlation (can happen with zero values)
            if np.isnan(corr):
                continue
            
            if corr > best_corr:
                best_corr = corr
                best_decay = decay
        
        print(f"Channel {channel}: Optimal decay rate = {best_decay:.2f}, "
              f"Correlation = {best_corr:.4f}")
        
        return best_decay, best_corr
    
    def _apply_adstock(self, series: pd.Series, decay_rate: float) -> pd.Series:
        """
        Apply adstock transformation to a series.
        
        Args:
            series: Original spend series
            decay_rate: Decay rate (between 0 and 1)
            
        Returns:
            Transformed series
        """
        result = series.copy()
        
        for i in range(1, len(series)):
            result.iloc[i] = series.iloc[i] + decay_rate * result.iloc[i-1]
            
        return result
    
    def prepare_data_for_modeling(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for modeling with train/test split and transformations.
        
        Returns:
            Dictionary with train and test DataFrames
        """
        if self.adstocked_data is None:
            raise ValueError("Adstock transformation not applied. Call apply_adstock_transformation() first.")
        
        print("Preparing data for modeling...")
        
        # Create datasets for different model types
        datasets = {}
        
        # Base dataset with adstocked variables
        modeling_data = self.adstocked_data.copy()
        
        # Add time-based features if 'week' column is date-like
        try:
            if not pd.api.types.is_datetime64_any_dtype(modeling_data['week']):
                modeling_data['week'] = pd.to_datetime(modeling_data['week'])
            
            modeling_data['month'] = modeling_data['week'].dt.month
            modeling_data['quarter'] = modeling_data['week'].dt.quarter
        except:
            # If 'week' is not date-like, skip adding time features
            print("Warning: Could not add time-based features.")
        
        # Train/test split
        split_idx = int(len(modeling_data) * self.config["mmm"]["train_test_split"])
        
        self.train_data = modeling_data.iloc[:split_idx].copy()
        self.test_data = modeling_data.iloc[split_idx:].copy()
        
        datasets["original"] = {
            "train": self.train_data,
            "test": self.test_data
        }
        
        # Create transformed datasets for different model types
        
        # Log-log transformation (multiplicative model)
        log_log_train = self.train_data.copy()
        log_log_test = self.test_data.copy()
        
        # Apply log transformation to sales and spend variables
        log_log_train['sales'] = np.log1p(log_log_train['sales'])
        log_log_test['sales'] = np.log1p(log_log_test['sales'])
        
        for channel in self.spend_columns:
            adstocked_col = f"{channel}_adstocked"
            # Add small epsilon to avoid log(0)
            log_log_train[adstocked_col] = np.log1p(log_log_train[adstocked_col])
            log_log_test[adstocked_col] = np.log1p(log_log_test[adstocked_col])
        
        datasets["log_log"] = {
            "train": log_log_train,
            "test": log_log_test
        }
        
        # Semi-log transformation
        semi_log_train = self.train_data.copy()
        semi_log_test = self.test_data.copy()
        
        # Apply log transformation only to sales
        semi_log_train['sales'] = np.log1p(semi_log_train['sales'])
        semi_log_test['sales'] = np.log1p(semi_log_test['sales'])
        
        datasets["semi_log"] = {
            "train": semi_log_train,
            "test": semi_log_test
        }
        
        # Save prepared datasets
        for model_type, data_dict in datasets.items():
            train_path = os.path.join(self.config["paths"]["outputs"]["eda"], 
                                    f"{model_type}_train.csv")
            test_path = os.path.join(self.config["paths"]["outputs"]["eda"], 
                                   f"{model_type}_test.csv")
            
            data_dict["train"].to_csv(train_path, index=False)
            data_dict["test"].to_csv(test_path, index=False)
        
        print("Data preparation completed")
        
        return datasets
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        print("Starting analysis pipeline...")
        
        # Load data
        self.load_data()
        
        # Perform EDA
        self.perform_eda()
        
        # Apply adstock transformation
        self.apply_adstock_transformation()
        
        # Prepare data for modeling
        prepared_data = self.prepare_data_for_modeling()
        
        # Compile all results
        results = {
            "eda": self.results.get("eda", {}),
            "adstock_params": self.adstock_params,
            "data_paths": {
                model_type: {
                    "train": os.path.join(self.config["paths"]["outputs"]["eda"], 
                                       f"{model_type}_train.csv"),
                    "test": os.path.join(self.config["paths"]["outputs"]["eda"], 
                                      f"{model_type}_test.csv")
                }
                for model_type in prepared_data.keys()
            }
        }
        
        # Save all results to a single JSON file
        with open(os.path.join(self.config["paths"]["outputs"]["eda"], 
                              "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=4, default=str)
        
        print("Analysis pipeline completed")
        
        return results


if __name__ == "__main__":
    # Test the analyst agent
    agent = AnalystAgent("config/agent_config.yaml")
    results = agent.run_analysis()
    print("Analysis results:", results)