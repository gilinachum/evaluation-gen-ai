"""
Script to download the MeetingBank dataset from Hugging Face.
The script I've created:

Uses the Hugging Face datasets library to download the MeetingBank dataset

Prints information about the dataset structure and available splits

Shows a sample from the training set to verify the data format

The dataset will be cached locally in the Hugging Face cache directory (typically ~/.cache/huggingface/datasets). The script provides a simple way to access the dataset programmatically for your meeting summarization task.
"""
from datasets import load_dataset
import os

def download_meetingbank():
    """
    Download the MeetingBank dataset and save it locally.
    """
    print("Downloading MeetingBank dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("huuuyeah/meetingbank")
    
    # Print dataset information
    print(f"Dataset structure: {dataset}")
    print(f"Available splits: {dataset.keys()}")
    
    # Print sample data
    if 'train' in dataset:
        print("\nSample from training set:")
        print(dataset['train'][0])
    
    print("\nDataset downloaded successfully!")
    
    return dataset

if __name__ == "__main__":
    dataset = download_meetingbank()