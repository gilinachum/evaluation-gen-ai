"""
Utility functions for loading and preprocessing the MeetingBank dataset.
"""
from datasets import load_dataset
import pandas as pd
import json

def load_meetingbank_dataset():
    """
    Load the MeetingBank dataset from Hugging Face.
    
    Returns:
        dataset: The loaded dataset object
    """
    return load_dataset("huuuyeah/meetingbank")

def get_test_samples(dataset, num_samples=2):
    """
    Get the first n samples from the test set.
    
    Args:
        dataset: The MeetingBank dataset
        num_samples: Number of samples to retrieve (default: 2)
        
    Returns:
        Dataset: A slice of the test dataset with num_samples items
    """
    if 'test' not in dataset:
        raise ValueError("Test split not found in the dataset")
    
    # Return the dataset slice directly, not individual samples
    return dataset['test'].select(range(num_samples))

def prepare_for_bedrock_evaluation(samples, include_model_responses=False):
    """
    Prepare samples for Bedrock evaluation.
    
    Args:
        samples: List of samples from the dataset
        include_model_responses: Whether to include empty modelResponses field (default: False)
        
    Returns:
        str: Path to the created jsonl file
    """
    # Format data according to Bedrock evaluation requirements
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-prompt-datasets-judge.html
    evaluation_data = []
    
    for sample in samples:
        # Extract transcript and summary
        transcript = sample['transcript']
        reference_summary = sample['summary']
        
        # Create prompt for summarization task
        prompt = f"Summarize the following meeting transcript:\n\n{transcript}"
        
        # Create evaluation record
        record = {
            "prompt": prompt,
            "referenceResponse": reference_summary,
            "category": "meeting_summarization"
        }
        
        # Add empty modelResponses field if requested
        if include_model_responses:
            record["modelResponses"] = []
        
        evaluation_data.append(record)
    
    # Save as JSONL file
    output_path = "./data/bedrock_evaluation_dataset.jsonl"
    with open(output_path, 'w') as f:
        for record in evaluation_data:
            f.write(json.dumps(record) + '\n')
    
    return output_path

def load_evaluation_dataset(dataset_path):
    """
    Load an evaluation dataset from a JSONL file.
    
    Args:
        dataset_path: Path to the dataset JSONL file
        
    Returns:
        list: List of evaluation records
    """
    with open(dataset_path, 'r') as f:
        return [json.loads(line) for line in f]