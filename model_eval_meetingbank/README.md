# MeetingBank Evaluation

This project demonstrates how to evaluate various models (both Amazon Bedrock and external models like Google's Gemini) on the MeetingBank dataset for meeting summarization tasks.

## Overview

The project includes:

1. Tools to download and prepare the MeetingBank dataset
2. Utilities for invoking models using Amazon Bedrock's Converse API and Google's Gemini API
3. Support for pre-generating model responses for evaluation
4. Integration with Amazon Bedrock's evaluation capabilities

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure AWS credentials with appropriate permissions for Amazon Bedrock and S3.

3. For Gemini models, set up Google Cloud credentials or API key as needed.

## Usage

### Standard Bedrock Evaluation

Run the `bedrock_evaluation.ipynb` notebook to:
- Load the MeetingBank dataset
- Prepare the dataset for Bedrock evaluation
- Create and run a Bedrock evaluation job
- Analyze and visualize the evaluation results

### Evaluation with Pre-generated Responses

For evaluating non-Bedrock models (like Gemini Flash), use the following workflow:
Run the `bedrock_evaluation_pregenerated.ipynb` notebook.

1. Prepare the dataset:
   ```
   python download_dataset.py
   ```

2. Generate model responses using the Converse API for Bedrock models:
   ```
   python pregenerate_responses.py --dataset ./data/bedrock_evaluation_dataset.jsonl --models "claude-3-haiku:us.anthropic.claude-3-haiku-20240307-v1:0" "nova-lite:us.amazon.nova-lite-v1:0"
   ```

   This will create separate files for each model, as Bedrock evaluation jobs support only one model response per prompt.

3. For Gemini models, use the dedicated script:
   ```
   python pregenerate_gemini_responses.py --dataset ./data/bedrock_evaluation_dataset.jsonl --model "gemini-2.0-flash" --project "your-gcp-project-id"
   ```

4. Run the appropriate notebook:
   - `bedrock_evaluation_pregenerated.ipynb`: For Bedrock models with pre-generated responses
   - `gemini_evaluation.ipynb`: For evaluating Gemini models

## Project Structure

- `data/`: Contains the MeetingBank dataset and prepared evaluation datasets
- `results/`: Contains evaluation results
- `utils/`: Utility functions for dataset preparation, model invocation, and evaluation
  - `bedrock_utils.py`: Utilities for Amazon Bedrock evaluation
  - `dataset_utils.py`: Utilities for loading and preprocessing the MeetingBank dataset
  - `external_model_utils.py`: Utilities for working with external (non-Bedrock) models
- `pregenerate_responses.py`: Script to pre-generate model responses using the Bedrock Converse API
- `pregenerate_gemini_responses.py`: Script to pre-generate model responses using Google's Gemini API
- `bedrock_evaluation.ipynb`: Notebook for standard Bedrock evaluation
- `bedrock_evaluation_pregenerated.ipynb`: Notebook for evaluation with pre-generated Bedrock model responses
- `gemini_evaluation.ipynb`: Notebook for evaluation with pre-generated Gemini model responses

## License

See the LICENSE file for details.