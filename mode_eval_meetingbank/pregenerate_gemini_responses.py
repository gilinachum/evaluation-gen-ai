#!/usr/bin/env python
"""
Script to pre-generate responses from Gemini models for the MeetingBank dataset.
"""
import os
import argparse
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility functions
from utils.dataset_utils import load_evaluation_dataset
from utils.external_model_utils import generate_gemini_responses

def main():
    """Main function to generate responses from Gemini models."""
    parser = argparse.ArgumentParser(description='Generate responses from Gemini models for MeetingBank dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the evaluation dataset JSONL file')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='Gemini model ID (default: gemini-2.0-flash)')
    parser.add_argument('--output', type=str, help='Path to save the output JSONL file')
    parser.add_argument('--project', type=str, help='Google Cloud project ID')
    parser.add_argument('--location', type=str, default='global', help='Google Cloud location (default: global)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature parameter (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum tokens to generate (default: 2048)')
    parser.add_argument('--top-p', type=float, default=0.7, help='Top-p parameter (default: 0.7)')
    parser.add_argument('--system-prompt', type=str, default='You are an expert meeting summarizer.', 
                        help='System prompt to use (default: "You are an expert meeting summarizer.")')
    
    args = parser.parse_args()
    
    # Generate responses
    try:
        output_path = generate_gemini_responses(
            dataset_path=args.dataset,
            model_id=args.model,
            output_path=args.output,
            system_prompt=args.system_prompt,
            project_id=args.project,
            location=args.location,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        )
        
        logger.info(f"Successfully generated responses using {args.model}")
        logger.info(f"Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        raise

if __name__ == "__main__":
    main()