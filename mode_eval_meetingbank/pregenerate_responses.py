"""
Script to pre-generate model responses for evaluation.

This script uses the Converse API to generate responses from various models
for a given dataset, and saves the responses in a format compatible with
Amazon Bedrock Evaluation.
"""
import os
import argparse
import logging
from datetime import datetime

# Import utility functions
from utils.dataset_utils import load_evaluation_dataset
from utils.bedrock_utils import generate_model_responses

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to pre-generate model responses.
    """
    parser = argparse.ArgumentParser(description='Pre-generate model responses for evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset JSONL file')
    parser.add_argument('--output-dir', type=str, help='Directory to save the output JSONL files')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--system-prompt', type=str, help='Optional system prompt to use')
    parser.add_argument('--models', type=str, nargs='+', required=True, 
                        help='Model IDs to use (format: name:model_id)')
    
    args = parser.parse_args()
    
    # Parse model IDs
    models = []
    for model_arg in args.models:
        parts = model_arg.split(':', 1)
        if len(parts) != 2:
            logger.error(f"Invalid model format: {model_arg}. Use name:model_id")
            return
        
        models.append({
            'name': parts[0],
            'model_id': parts[1]
        })
    
    logger.info(f"Using models: {models}")
    
    # Set output directory if not provided
    output_dir = args.output_dir
    if not output_dir:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = f"./data/responses_{timestamp}"
    
    # Generate responses
    try:
        result_paths = generate_model_responses(
            dataset_path=args.dataset,
            models=models,
            output_dir=output_dir,
            system_prompt=args.system_prompt,
            region=args.region
        )
        
        logger.info("Successfully generated responses. Results saved to:")
        for model_name, path in result_paths.items():
            logger.info(f"  - {model_name}: {path}")
        
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        raise

if __name__ == "__main__":
    main()