"""
Utility functions for Amazon Bedrock evaluation.
"""
import boto3
import json
import time
import os
from botocore.exceptions import ClientError

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
    model_id,
    region="us-east-1"
):
    """
    Create a Bedrock model evaluation job.
    
    Args:
        job_name: Name for the evaluation job
        dataset_s3_uri: S3 URI of the evaluation dataset
        output_s3_uri: S3 URI for the output
        model_ids: List of model IDs to evaluate
        region: AWS region (default: us-east-1)
        
    Returns:
        str: Job ID of the created evaluation job
    """
    bedrock = boto3.client('bedrock', region_name=region)
    
    # Create model configs according to API documentation
    model_configs = []
    model_configs.append({
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
    })
    
    # Create evaluation job
    response = bedrock.create_evaluation_job(
        jobName=job_name,
        jobDescription="MeetingBank summarization evaluation",
        roleArn=os.environ.get("BEDROCK_ROLE_ARN"),  # Role with necessary permissions
        outputDataConfig={
            "s3Uri": output_s3_uri
        },
        evaluationConfig={
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
                            "Builtin.Relevance",
                            "Builtin.Correctness",
                            "Builtin.Completeness",
                            "Builtin.Coherence",
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
        },
        inferenceConfig={
            "models": model_configs
        }
    )
    
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