"""
Simple script to run the MMM system with data from a file.
"""

import argparse
from main import run_mmm_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMM system with input data")
    parser.add_argument("--input-file", type=str, required=True,
                      help="Path to the input data file (CSV or space-separated)")
    
    args = parser.parse_args()
    
    # Run the pipeline with the specified data file
    results = run_mmm_pipeline(data_path=args.input_file)
    
    print("\nAnalysis and modeling completed successfully.")
    print("Run the dashboard with: streamlit run ui/app.py")