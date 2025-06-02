"""
Utility functions for invoking external (non-Bedrock) models.
"""
import json
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def format_for_bedrock_evaluation(
    dataset_path: str,
    responses_path: str,
    output_path: Optional[str] = None,
    model_name: str = "external_model"
) -> str:
    """
    Format external model responses for Bedrock evaluation.
    
    Args:
        dataset_path: Path to the original dataset JSONL file
        responses_path: Path to the file containing model responses
        output_path: Path to save the formatted output (default: None)
        model_name: Name to identify the model in the results (default: "external_model")
        
    Returns:
        str: Path to the formatted output file
    """
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Load the responses
    with open(responses_path, 'r') as f:
        responses = [json.loads(line) for line in f]
    
    # If output_path is not provided, create a default one
    if not output_path:
        output_path = dataset_path.replace('.jsonl', f'_with_{model_name}_responses.jsonl')
    
    # Match responses to prompts and format for Bedrock evaluation
    for i, record in enumerate(dataset):
        if i < len(responses):
            response = responses[i]
            
            # Initialize modelResponses if not present
            if 'modelResponses' not in record:
                record['modelResponses'] = []
            
            # Add the response
            record['modelResponses'].append({
                "response": response['response'],
                "modelIdentifier": model_name
            })
    
    # Save the formatted dataset
    with open(output_path, 'w') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Formatted dataset saved to {output_path}")
    return output_path

def merge_model_responses(
    dataset_paths: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Merge multiple datasets with model responses into a single dataset.
    
    Args:
        dataset_paths: List of paths to datasets with model responses
        output_path: Path to save the merged dataset (default: None)
        
    Returns:
        str: Path to the merged dataset
    """
    # Load all datasets
    datasets = []
    for path in dataset_paths:
        with open(path, 'r') as f:
            datasets.append([json.loads(line) for line in f])
    
    # If output_path is not provided, create a default one
    if not output_path:
        output_path = dataset_paths[0].replace('.jsonl', '_merged.jsonl')
    
    # Use the first dataset as the base
    merged_dataset = datasets[0]
    
    # Merge model responses from other datasets
    for dataset in datasets[1:]:
        for i, record in enumerate(dataset):
            if i < len(merged_dataset) and 'modelResponses' in record:
                # Add model responses that don't already exist
                for model_response in record['modelResponses']:
                    model_id = model_response['modelIdentifier']
                    
                    # Check if this model response already exists
                    exists = any(mr['modelIdentifier'] == model_id 
                                for mr in merged_dataset[i].get('modelResponses', []))
                    
                    if not exists:
                        if 'modelResponses' not in merged_dataset[i]:
                            merged_dataset[i]['modelResponses'] = []
                        merged_dataset[i]['modelResponses'].append(model_response)
    
    # Save the merged dataset
    with open(output_path, 'w') as f:
        for record in merged_dataset:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Merged dataset saved to {output_path}")
    return output_path

# Example function for invoking Gemini Flash (placeholder)
# You would need to implement this based on the actual Gemini API
def invoke_gemini_flash(
    prompts: List[str],
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 2048
) -> List[Dict[str, Any]]:
    """
    Invoke Gemini Flash model with a list of prompts.
    
    Args:
        prompts: List of prompts to send to the model
        api_key: Gemini API key
        temperature: Temperature parameter (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 2048)
        
    Returns:
        List[Dict[str, Any]]: List of response dictionaries
    """
    responses = []
    
    # This is a placeholder implementation
    # You would need to replace this with actual Gemini API calls
    for prompt in prompts:
        logger.info(f"Invoking Gemini Flash for prompt: {prompt[:50]}...")
        
        # Example API call (replace with actual implementation)
        # response = requests.post(
        #     "https://api.gemini.com/v1/models/gemini-flash:generateContent",
        #     headers={"Authorization": f"Bearer {api_key}"},
        #     json={
        #         "contents": [{"parts": [{"text": prompt}]}],
        #         "generationConfig": {
        #             "temperature": temperature,
        #             "maxOutputTokens": max_tokens
        #         }
        #     }
        # )
        # response_json = response.json()
        # response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
        
        # Placeholder response
        response_text = f"This is a placeholder response for Gemini Flash. Implement the actual API call."
        
        responses.append({
            "response": response_text,
            "modelIdentifier": "gemini-flash"
        })
    
    return responses