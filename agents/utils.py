"""
Utility functions for Market Mix Modeling agents.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    converted_data = convert_numpy(data)
    
    # Save to file
    with open(file_path, "w") as f:
        json.dump(converted_data, f, indent=4)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data


def find_optimal_adstock_decay(
    sales: pd.Series, 
    spend: pd.Series,
    decay_range: List[float] = [0.1, 0.9],
    num_points: int = 20
) -> Tuple[float, float]:
    """
    Find optimal adstock decay rate that maximizes correlation with sales.
    
    Args:
        sales: Sales data
        spend: Marketing spend data
        decay_range: Range of decay rates to try [min, max]
        num_points: Number of decay rates to try
        
    Returns:
        Tuple of (best_decay_rate, best_correlation)
    """
    # Check if channel has any non-zero values
    if spend.sum() == 0 or spend.nunique() <= 1:
        print(f"Channel has no variation or only zero values. Using default decay rate.")
        return decay_range[0], 0
    
    decay_values = np.linspace(decay_range[0], decay_range[1], num_points)
    best_decay = decay_range[0]
    best_corr = 0
    
    for decay in decay_values:
        adstocked = apply_adstock(spend, decay)
        
        # Handle case where adstocked values have no variation
        if np.std(adstocked) == 0:
            continue
            
        corr = abs(np.corrcoef(adstocked, sales)[0, 1])
        
        # Handle NaN correlation (can happen with zero values)
        if np.isnan(corr):
            continue
        
        if corr > best_corr:
            best_corr = corr
            best_decay = decay
    
    return best_decay, best_corr


def apply_adstock(series: pd.Series, decay_rate: float) -> pd.Series:
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


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase and fix common naming issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Convert to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Remove whitespace and special characters
    df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Standardize common column names
    rename_map = {
        'date': 'week',
        'time': 'week',
        'period': 'week',
        'month': 'week',
        'revenue': 'sales',
        'income': 'sales',
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    # For numeric columns, replace NaN with 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # For non-numeric columns, forward-fill and then backward-fill
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        df[non_numeric_cols] = df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    return df