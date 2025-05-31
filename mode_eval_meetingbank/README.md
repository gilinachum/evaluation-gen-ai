# MeetingBank Model Evaluation

This project demonstrates how to evaluate the accuracy of different Amazon Bedrock models (Nova.lite and Nova.pro) on the MeetingBank dataset for meeting summarization tasks.

## Dataset

We use the [MeetingBank dataset](https://huggingface.co/datasets/huuuyeah/meetingbank) which provides meeting transcripts and matching human-created summaries. This makes it ideal for evaluating summarization capabilities of large language models.

## Project Structure

- `download_dataset.py` - Simple script to download the MeetingBank dataset
- `bedrock_evaluation.ipynb` - Main notebook for running model evaluations
- `requirements.txt` - Dependencies for the project
- `utils/` - Utility modules for dataset handling and Bedrock evaluation

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the MeetingBank dataset:
   ```bash
   python download_dataset.py
   ```

3. Open and run the evaluation notebook:
   ```bash
   jupyter notebook bedrock_evaluation.ipynb
   ```

4. Follow the steps in the notebook to:
   - Load the dataset
   - Prepare it for Bedrock evaluation
   - Configure and run evaluation jobs
   - Analyze and visualize the results

## AWS Configuration

Before running the evaluation, you need to:

1. Configure AWS credentials with access to Amazon Bedrock and S3
2. Create an IAM role with permissions for Bedrock evaluation jobs
3. Update the `BEDROCK_ROLE_ARN` in the notebook with your role ARN

## Models Evaluated

TBD

The evaluation uses Bedrock's built-in evaluators to assess:
- Relevance
- Accuracy
- Coherence
- Conciseness