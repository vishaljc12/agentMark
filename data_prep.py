"""
Data preparation script to handle various input formats for Market Mix Modeling.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import re


def prepare_data(input_text=None, input_file=None, output_path="data/monthly_data.csv"):
    """
    Prepare data for Market Mix Modeling, handling various input formats.
    
    Args:
        input_text: Raw text data
        input_file: Path to file containing data
        output_path: Path to save the prepared CSV file
    
    Returns:
        Path to the prepared CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try to load data in different formats
    if input_file:
        print(f"Reading data from file: {input_file}")
        # Try to infer the file type and load accordingly
        if input_file.endswith('.csv'):
            try:
                # Try standard CSV first
                df = pd.read_csv(input_file)
                print("Loaded as standard CSV file")
            except:
                # Try with different delimiters
                try:
                    df = pd.read_csv(input_file, sep=None, engine='python')
                    print("Loaded CSV with automatic delimiter detection")
                except Exception as e:
                    print(f"Error loading CSV: {str(e)}")
                    # Try to parse as space/tab separated file
                    try:
                        with open(input_file, 'r') as f:
                            content = f.read()
                        df = _parse_text_data(content)
                        print("Parsed as space/tab separated file")
                    except Exception as e2:
                        raise ValueError(f"Failed to parse file: {str(e2)}")
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            try:
                df = pd.read_excel(input_file)
                print("Loaded as Excel file")
            except Exception as e:
                raise ValueError(f"Error loading Excel file: {str(e)}")
        else:
            # Try to parse as text file with space/tab separation
            try:
                with open(input_file, 'r') as f:
                    content = f.read()
                df = _parse_text_data(content)
                print("Parsed as space/tab separated file")
            except Exception as e:
                raise ValueError(f"Failed to parse file: {str(e)}")
    elif input_text:
        print("Parsing input text data")
        df = _parse_text_data(input_text)
    else:
        raise ValueError("Either input_text or input_file must be provided")
    
    # Clean and standardize the data
    df = _clean_data(df)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Data prepared and saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return output_path


def _parse_text_data(text_data):
    """
    Parse text data that could be space, tab, or comma separated.
    
    Args:
        text_data: Text data to parse
        
    Returns:
        Pandas DataFrame
    """
    lines = text_data.strip().split('\n')
    
    # Try to detect delimiter
    first_line = lines[0]
    if ',' in first_line:
        delimiter = ','
    elif '\t' in first_line:
        delimiter = '\t'
    else:
        delimiter = None  # Space or multiple spaces
    
    if delimiter:
        # Use CSV reader for comma or tab delimited data
        reader = csv.reader(lines, delimiter=delimiter)
        rows = list(reader)
    else:
        # Handle space-delimited data
        rows = []
        for line in lines:
            # Split by any number of spaces
            row = re.split(r'\s+', line.strip())
            rows.append(row)
    
    # Extract header and data
    header = rows[0]
    data = rows[1:]
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns
    for col in df.columns:
        if col != 'Week':  # Don't convert date column
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # Keep as string if conversion fails
    
    return df


def _clean_data(df):
    """
    Clean and standardize the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Remove unnamed or empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Check for date/time column
    date_cols = [col for col in df.columns if any(word in col.lower() 
                                                for word in ['date', 'week', 'month', 'period', 'time'])]
    
    # If found, rename to 'week' for consistency
    if date_cols and 'week' not in df.columns:
        df = df.rename(columns={date_cols[0]: 'week'})
    
    # Ensure 'week' column exists (use first column if not found)
    if 'week' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'week'})
    
    # Convert 'week' to datetime if possible
    if 'week' in df.columns:
        try:
            df['week'] = pd.to_datetime(df['week'])
        except:
            print("Warning: Unable to convert 'week' column to datetime")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for Market Mix Modeling")
    parser.add_argument("--input-file", type=str, help="Path to input data file")
    parser.add_argument("--output-file", type=str, default="data/monthly_data.csv",
                      help="Path to save prepared data")
    
    args = parser.parse_args()
    
    if args.input_file:
        prepare_data(input_file=args.input_file, output_path=args.output_file)
    else:
        print("Please provide an input file path using --input-file")