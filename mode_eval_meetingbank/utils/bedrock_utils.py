"""
Utility functions for Amazon Bedrock evaluation.
"""
import boto3
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def apply_cors_if_not_exists(bucket_name, region="us-east-1"):
    """
    Apply CORS configuration to an S3 bucket if it doesn't exist.
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (default: us-east-1)
        
    Returns:
        bool: True if CORS was applied, False if it already existed
    """
    s3 = boto3.client('s3', region_name=region)
    
    try:
        # Check if CORS configuration already exists
        cors = s3.get_bucket_cors(Bucket=bucket_name)
        print(f"CORS configuration already exists for bucket {bucket_name}")
        return False
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchCORSConfiguration':
            # Apply CORS configuration
            cors_configuration = {
                'CORSRules': [
                    {
                        'AllowedHeaders': ['*'],
                        'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
                        'AllowedOrigins': ['*'],
                        'ExposeHeaders': ['Access-Control-Allow-Origin']
                    }
                ]
            }
            s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_configuration)
            print(f"Applied CORS configuration to bucket {bucket_name}")
            return True
        else:
            # Re-raise if it's a different error
            raise

def create_s3_bucket_if_not_exists(bucket_name, region="us-east-1"):
    """
    Create an S3 bucket if it doesn't exist.
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (default: us-east-1)
        
    Returns:
        str: Bucket name
    """
    s3 = boto3.client('s3', region_name=region)
    
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} already exists")
    except ClientError:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"Created bucket {bucket_name}")
    
    return bucket_name

def upload_to_s3(file_path, bucket_name, s3_key, region="us-east-1"):
    """
    Upload a file to S3.
    
    Args:
        file_path: Path to the local file
        bucket_name: Name of the S3 bucket
        s3_key: S3 object key
        region: AWS region (default: us-east-1)
        
    Returns:
        str: S3 URI of the uploaded file
    """
    s3 = boto3.client('s3', region_name=region)
    s3.upload_file(file_path, bucket_name, s3_key)
    return f"s3://{bucket_name}/{s3_key}"

def create_evaluation_job(
    job_name,
    dataset_s3_uri,
    output_s3_uri,
    model_id=None, # Needed even for pregenerated
    region="us-east-1",
    use_pregenerated_responses=False
):
    """
    Create a Bedrock model evaluation job.
    
    Args:
        job_name: Name for the evaluation job
        dataset_s3_uri: S3 URI of the evaluation dataset
        output_s3_uri: S3 URI for the output
        model_id: Model ID to evaluate (not needed if use_pregenerated_responses=True)
        region: AWS region (default: us-east-1)
        use_pregenerated_responses: Whether to use pre-generated responses (default: False)
        
    Returns:
        str: Job ID of the created evaluation job
    """
    bedrock = boto3.client('bedrock', region_name=region)
    
    # Create evaluation job configuration
    job_config = {
        "jobName": job_name,
        "jobDescription": "MeetingBank summarization evaluation",
        "roleArn": os.environ.get("BEDROCK_ROLE_ARN"),  # Role with necessary permissions
        "outputDataConfig": {
            "s3Uri": output_s3_uri
        },
        "evaluationConfig": {
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": "General",
                        "dataset": {
                            "name": "meetingbank_dataset",
                            "datasetLocation": {
                                "s3Uri": dataset_s3_uri
                            }
                        },
                        "metricNames": [
                            #"Builtin.Relevance",
                            "Builtin.Correctness",
                            "Builtin.Completeness",
                            #"Builtin.Coherence",
                        ]
                    }
                ],
                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {
                            "modelIdentifier": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
                        }
                    ]
                }
            }
        }
    }

    if use_pregenerated_responses:
        job_config["inferenceConfig"] = {
                "models": [
                    {
                        "precomputedInferenceSource": {
                            "inferenceSourceIdentifier": model_id
                        }
                    }
                ]
            }
    
    # Add inference config only if not using pre-generated responses
    else:
        if not model_id:
            raise ValueError("model_id must be provided when use_pregenerated_responses=False")
        
        model_configs = [{
            "bedrockModel": {
                "modelIdentifier": model_id,
                "inferenceParams": json.dumps({
                    "inferenceConfig" : {
                        "temperature": 0.0,
                        "topP": 0.7,
                        "maxTokens": 2048
                    }
                })
            }
        }]
        
        job_config["inferenceConfig"] = {
            "models": model_configs
        }
    
    # Create evaluation job
    response = bedrock.create_evaluation_job(**job_config)
    
    return response['jobArn']

def get_evaluation_job_status(job_arn, region="us-east-1"):
    """
    Get the status of an evaluation job.
    
    Args:
        job_arn: ARN of the evaluation job
        region: AWS region (default: us-east-1)
        
    Returns:
        str: Status of the evaluation job
    """
    bedrock = boto3.client('bedrock', region_name=region)
    response = bedrock.get_evaluation_job(jobIdentifier=job_arn)
    return response['status']

def wait_for_job_completion(job_arn, region="us-east-1", poll_interval=60):
    """
    Wait for an evaluation job to complete.
    
    Args:
        job_arn: ARN of the evaluation job
        region: AWS region (default: us-east-1)
        poll_interval: Polling interval in seconds (default: 60)
        
    Returns:
        dict: Job details
    """
    bedrock = boto3.client('bedrock', region_name=region)
    
    while True:
        response = bedrock.get_evaluation_job(jobIdentifier=job_arn)
        status = response['status']
        
        if status.upper() in ['COMPLETED', 'FAILED', 'STOPPED']:
            return response
        
        print(f"Job status: {status}. Waiting {poll_interval} seconds...")
        time.sleep(poll_interval)

def download_evaluation_results(s3_uri, local_dir, region="us-east-1"):
    """
    Recursively discover and download evaluation result files from S3 that end with _output.jsonl.
    
    Args:
        s3_uri: S3 URI of the evaluation results directory (e.g., s3://bucket/path/to/results/)
        local_dir: Local directory to save the downloaded files
        region: AWS region (default: us-east-1)
        
    Returns:
        list: Paths to the downloaded result files
    """
    s3 = boto3.client('s3', region_name=region)
    
    # Parse S3 URI
    bucket_name = s3_uri.split('/')[2]
    prefix = '/'.join(s3_uri.split('/')[3:])
    prefix = prefix.replace("//", "/")
    
    # Ensure prefix ends with a slash
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    print(f'prefix={prefix}')
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects in the bucket with the given prefix
    downloaded_files = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Download only files ending with _output.jsonl
            if key.endswith('_output.jsonl'):
                # Create local file path
                relative_path = key[len(prefix):] if prefix else key
                local_file_path = os.path.join(local_dir, "output.jsonl")
                
                # # Create directory structure if needed
                # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the file
                print(f"Downloading {key} to {local_file_path}")
                s3.download_file(bucket_name, key, local_file_path)
                downloaded_files.append(local_file_path)
    
    return downloaded_files

def invoke_model_with_converse(
    prompt,
    model_id,
    model_name,
    system_prompt=None,
    temperature=0.0,
    max_tokens=2048,
    top_p=0.7,
    region="us-east-1"
):
    """
    Invoke a model using the Converse API.
    
    Args:
        prompt: The prompt to send to the model
        model_id: The model ID to use
        model_name: A name to identify the model in the results
        system_prompt: Optional system prompt to use
        temperature: Temperature parameter for inference
        max_tokens: Maximum tokens to generate
        top_p: Top-p parameter for inference
        region: AWS region (default: us-east-1)
        
    Returns:
        str: The model's response
    """
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)
    
    logger.info(f"Generating response with model {model_id}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    
    # Prepare system prompt if provided
    system_prompts = [{"text": ""}]
    if system_prompt:
        system_prompts = [{"text": system_prompt}]
    
    # Inference parameters
    inference_config = {
        "temperature": temperature,
        "topP": top_p,
        "maxTokens": max_tokens
    }
    
    try:
        # Send the request
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config
        )
        
        # Log token usage
        token_usage = response.get('usage', {})
        logger.debug(f"Input tokens: {token_usage.get('inputTokens', 'N/A')}")
        logger.debug(f"Output tokens: {token_usage.get('outputTokens', 'N/A')}")
        logger.debug(f"Total tokens: {token_usage.get('totalTokens', 'N/A')}")
        logger.debug(f"Stop reason: {response.get('stopReason', 'N/A')}")
        
        # Extract the response text
        output_message = response['output']['message']
        response_text = ""
        for content in output_message['content']:
            if content.get('text'):
                response_text += content['text']
        
        return response_text
    
    except ClientError as e:
        logger.error(f"Error invoking model {model_id}: {e}")
        raise

def generate_model_responses(
    dataset_path,
    models,
    output_dir=None,
    system_prompt=None,
    region="us-east-1"
):
    """
    Generate responses for all prompts in the dataset using the specified models.
    Creates a separate file for each model.
    
    Args:
        dataset_path: Path to the dataset JSONL file
        models: List of dictionaries with 'name' and 'model_id' keys
        output_dir: Directory to save the output JSONL files (default: None)
        system_prompt: Optional system prompt to use (default: None)
        region: AWS region (default: us-east-1)
        
    Returns:
        dict: Dictionary mapping model names to their output file paths
    """
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # If output_dir is not provided, create a default one
    if not output_dir:
        output_dir = os.path.dirname(dataset_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store output paths for each model
    output_paths = {}
    
    # Process each model separately
    for model in models:
        model_name = model['name']
        model_id = model['model_id']
        
        # Create a copy of the dataset for this model
        model_dataset = []
        for record in dataset:
            model_record = record.copy()
            # Initialize empty modelResponses array
            model_record['modelResponses'] = []
            model_dataset.append(model_record)
        
        # Generate output path for this model
        model_output_path = os.path.join(
            output_dir, 
            f"{os.path.basename(dataset_path).replace('.jsonl', '')}_{model_name}.jsonl"
        )
        output_paths[model_name] = model_output_path
        
        logger.info(f"Generating responses for model {model_name}")
        
        # Process each prompt for this model
        for i, record in enumerate(model_dataset):
            prompt = record['prompt']
            logger.info(f"Processing prompt {i+1}/{len(model_dataset)} for model {model_name}")
            
            try:
                response_text = invoke_model_with_converse(
                    prompt=prompt,
                    model_id=model_id,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    region=region
                )
                
                # Add the response to the record
                record['modelResponses'] = [{
                    "response": response_text,
                    "modelIdentifier": model_name
                }]
                
                # Save after each response to avoid losing progress
                with open(model_output_path, 'w') as f:
                    for r in model_dataset:
                        f.write(json.dumps(r) + '\n')
                
            except Exception as e:
                logger.error(f"Error generating response for model {model_name}, prompt {i+1}: {e}")
        
        logger.info(f"Dataset with {model_name} responses saved to {model_output_path}")
    
    return output_paths

def analyze_evaluation_results(results_dir):
    """
    Analyze evaluation results from JSONL files.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        pd.DataFrame: DataFrame with metrics for all models
    """
    import os
    import glob
    
    # Find all output.jsonl files
    result_files = glob.glob(os.path.join(results_dir, '*/output.jsonl'))
    
    all_metrics = {}
    
    for file_path in result_files:
        # Extract model name from directory name
        model_name = os.path.basename(os.path.dirname(file_path))
        model_name = model_name.split('-')[1] if '-' in model_name else model_name
        
        # Load results from JSONL file
        results = []
        with open(file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        # Extract metrics
        metrics = {}
        for result in results:
            if 'automatedEvaluationResult' in result and 'scores' in result['automatedEvaluationResult']:
                scores = result['automatedEvaluationResult']['scores']
                for score in scores:
                    metric_name = score['metricName'].replace('Builtin.', '')
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(score['result'])
        
        # Calculate average for each metric
        avg_metrics = {}
        for metric, values in metrics.items():
            avg_metrics[metric] = sum(values) / len(values) if values else 0
        
        all_metrics[model_name] = avg_metrics
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    return df

def visualize_evaluation_results(df, output_path=None):
    """
    Visualize metrics for all models.
    
    Args:
        df: DataFrame with metrics
        output_path: Path to save the visualization (optional)
    """
    # Create a bar chart
    ax = df.plot(kind='bar', figsize=(12, 8))
    ax.set_title('Model Evaluation Scores on MeetingBank Dataset')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)  # Scores are typically between 0 and 1
    plt.legend(title='Models')
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path)
    
    return ax