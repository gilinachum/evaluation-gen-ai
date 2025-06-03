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

def invoke_gemini_model(
    prompts: List[str],
    model_id: str = "gemini-2.0-flash",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 0.7,
    project_id: Optional[str] = None,
    location: str = "global"
) -> List[Dict[str, Any]]:
    """
    Invoke Gemini model with a list of prompts using Google's Generative AI SDK.
    
    Args:
        prompts: List of prompts to send to the model
        model_id: Gemini model ID (default: "gemini-2.0-flash")
        system_prompt: Optional system prompt to use (default: None)
        temperature: Temperature parameter (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 2048)
        top_p: Top-p parameter (default: 0.7)
        project_id: Google Cloud project ID (default: None)
        location: Google Cloud location (default: "global")
        
    Returns:
        List[Dict[str, Any]]: List of response dictionaries
    """
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError:
        logger.error("Google Generative AI library not installed. Install with: pip install google-generativeai")
        raise
    
    responses = []
    
    try:
        # Initialize the Gemini client
        if project_id:
            client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            # Use API key if available in environment
            client = genai.Client()
        
        logger.info(f"Initialized Gemini client for model {model_id}")
        
        # Process each prompt
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)} for model {model_id}")
            
            try:
                # Create configuration
                config = GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                )
                
                # Add system instruction if provided
                if system_prompt:
                    config.system_instruction = system_prompt
                
                # Generate content
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config
                )
                
                # Extract response text
                response_text = response.text
                
                sanitized_model_name = model_id.replace('.', '-').replace(':', '-')
                responses.append({
                    "response": response_text,
                    "modelIdentifier": sanitized_model_name
                })
                
                logger.info(f"Successfully generated response for prompt {i+1}")
                
            except Exception as e:
                logger.error(f"Error generating response for prompt {i+1}: {e}")
                # Add an error message as the response
                responses.append({
                    "response": f"Error generating response: {str(e)}",
                    "modelIdentifier": sanitized_model_name
                })
    
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {e}")
        # Return error responses for all prompts
        responses = [{"response": f"Error initializing Gemini client: {str(e)}", "modelIdentifier": model_id} for _ in prompts]
    
    return responses

def generate_gemini_responses(
    dataset_path: str,
    model_id: str = "gemini-2.0-flash",
    output_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    project_id: Optional[str] = None,
    location: str = "global",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 0.7
) -> str:
    """
    Generate responses for all prompts in the dataset using Gemini model.
    
    Args:
        dataset_path: Path to the dataset JSONL file
        model_id: Gemini model ID (default: "gemini-2.0-flash")
        output_path: Path to save the output JSONL file (default: None)
        system_prompt: Optional system prompt to use (default: None)
        project_id: Google Cloud project ID (default: None)
        location: Google Cloud location (default: "global")
        temperature: Temperature parameter (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 2048)
        top_p: Top-p parameter (default: 0.7)
        
    Returns:
        str: Path to the output file
    """
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Extract prompts
    prompts = [record['prompt'] for record in dataset]
    
    # Generate responses
    logger.info(f"Generating responses for {len(prompts)} prompts using Gemini model {model_id}")
    responses = invoke_gemini_model(
        prompts=prompts,
        model_id=model_id,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        project_id=project_id,
        location=location
    )
    
    # Create a copy of the dataset with responses
    model_dataset = []
    for i, record in enumerate(dataset):
        model_record = record.copy()
        if i < len(responses):
            model_record['modelResponses'] = [responses[i]]
        else:
            model_record['modelResponses'] = []
        model_dataset.append(model_record)
    
    # Generate output path if not provided
    if not output_path:
        model_name = model_id.replace('.', '-').replace(':', '-')
        output_path = dataset_path.replace('.jsonl', f'_{model_name}.jsonl')
    
    # Save the dataset with responses
    with open(output_path, 'w') as f:
        for record in model_dataset:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Dataset with {model_id} responses saved to {output_path}")
    return output_path