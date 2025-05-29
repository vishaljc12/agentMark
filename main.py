"""
Main script to run the Market Mix Modeling (MMM) multi-agent system.
"""

import os
import yaml
import json
import argparse
import autogen
from agents.analyst_agent import AnalystAgent
from agents.modeling_agent import ModelingAgent
from agents.ui_agent import UIAgent
from typing import Dict, Any, Optional
from data_prep import prepare_data


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dictionary with configuration
    """
    with open("config/agent_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("Loaded config:", config)
    
    return config


def setup_llm_config() -> Dict[str, Any]:
    """
    Set up LLM configuration for agents.
    
    Returns:
        Dictionary with LLM configuration
    """
    # Load Azure OpenAI configuration
    try:
        with open("config/config.yaml", "r") as f:
            azure_config = yaml.safe_load(f)
        
        azure_openai = azure_config.get("azure_openai", {})
        
        # Check if API key is provided
        if not azure_openai.get("api_key"):
            print("Warning: Azure OpenAI API key not found in config.yaml")
            return None
        
        # Configure LLM
        llm_config = {
            "config_list": [{
                "model": azure_openai.get("deployment_name", "gpt-4o"),
                "api_key": azure_openai.get("api_key", ""),
                "api_base": azure_openai.get("endpoint", ""),
                "api_type": "azure",
                "api_version": azure_openai.get("api_version", "2024-08-01-preview"),
                "deployment_id": azure_openai.get("deployment_name", "gpt-4o"),
            }],
            "temperature": 0.1,
            "timeout": 600,
            "cache_seed": 42,
        }
        
        return llm_config
    except Exception as e:
        print(f"Warning: Could not load LLM config: {str(e)}")
        return None


def run_mmm_pipeline(data_path: str = None, data_text: str = None, optimize_budget: bool = True) -> Dict[str, Any]:
    """
    Run the full MMM pipeline with all agents.
    
    Args:
        data_path: Path to the input data file
        data_text: Raw data text (alternative to data_path)
        optimize_budget: Whether to perform budget optimization
        
    Returns:
        Dictionary with results
    """
    print("Starting Market Mix Modeling (MMM) pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Setup LLM configuration
    llm_config = setup_llm_config()
    
    # Create output directories
    for path in config["paths"]["outputs"].values():
        os.makedirs(path, exist_ok=True)
    
    # Prepare data if raw text is provided
    if data_text:
        data_path = prepare_data(input_text=data_text)
        print(f"Data prepared and saved to {data_path}")
    elif not data_path:
        data_path = config["paths"]["data"]
    else:
        # If data_path provided but not in the default location, copy to default
        if data_path != config["paths"]["data"]:
            processed_path = prepare_data(input_file=data_path, output_path=config["paths"]["data"])
            data_path = processed_path
    
    # Step 1: Run Analyst Agent
    print("\n=== Running Analyst Agent ===")
    analyst_agent = AnalystAgent("config/agent_config.yaml", llm_config)
    analysis_results = analyst_agent.run_analysis()
    
    # Step 2: Run Modeling Agent
    print("\n=== Running Modeling Agent ===")
    modeling_agent = ModelingAgent(
        "config/agent_config.yaml", 
        "outputs/eda/analysis_results.json",
        llm_config
    )
    modeling_results = modeling_agent.run_modeling(optimize_budget=optimize_budget)
    
    # Step 3: Run UI Agent
    print("\n=== Running UI Agent ===")
    ui_agent = UIAgent(
        "config/agent_config.yaml",
        "outputs/eda/analysis_results.json",
        "outputs/models/modeling_results.json",
        llm_config
    )
    app_path = ui_agent.run()
    
    print(f"\n=== MMM Pipeline Completed ===")
    print(f"Dashboard created at: {app_path}")
    print("To run the dashboard, navigate to the project directory and execute:")
    print("streamlit run ui/app.py")
    
    # Compile all results
    results = {
        "analysis": analysis_results,
        "modeling": modeling_results,
        "dashboard_path": app_path
    }
    
    return results


def run_budget_optimization(total_budget: float) -> Dict[str, Any]:
    """
    Run only the budget optimization with a new total budget.
    
    Args:
        total_budget: New total budget to allocate
        
    Returns:
        Dictionary with optimization results
    """
    print(f"Running budget optimization for ${total_budget}...")
    
    # Setup LLM configuration
    llm_config = setup_llm_config()
    
    # Run Modeling Agent for optimization only
    modeling_agent = ModelingAgent(
        "config/agent_config.yaml", 
        "outputs/eda/analysis_results.json",
        llm_config
    )
    
    try:
        # Load models first
        modeling_agent.load_prepared_data()
        modeling_agent.build_models()
        modeling_agent.calculate_contributions()
        
        # Run optimization with new budget
        optimization_results = modeling_agent.optimize_budget(total_budget)
        
        print("Budget optimization completed")
        
        return optimization_results
    except Exception as e:
        print(f"Error during budget optimization: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Mix Modeling Multi-Agent System")
    parser.add_argument("--data", type=str, default=None, 
                        help="Path to input data file")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to file containing raw data (space-separated)")
    parser.add_argument("--optimize-only", action="store_true", 
                        help="Run only budget optimization")
    parser.add_argument("--budget", type=float, 
                        help="Total budget for optimization")
    
    args = parser.parse_args()
    
    if args.optimize_only and args.budget:
        # Run only budget optimization
        results = run_budget_optimization(args.budget)
        print("Optimization results:", json.dumps(results, indent=2))
    else:
        # Run full pipeline
        if args.data_file:
            # Read raw data from file
            with open(args.data_file, 'r') as f:
                data_text = f.read()
            results = run_mmm_pipeline(data_text=data_text)
        else:
            # Use specified data path or default
            results = run_mmm_pipeline(data_path=args.data)