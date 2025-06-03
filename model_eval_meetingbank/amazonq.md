# Bedrock Evaluation on MeetingBank Dataset

## Objective
Create a notebook to evaluate Nova.lite and Nova.pro models on the MeetingBank dataset using Amazon Bedrock's evaluation capabilities.

## Project Structure - keep it uptodate from time to time.
```
mode_eval_meetingbank/
├── README.md                   # Project overview and instructions
├── requirements.txt            # Dependencies for the project
├── download_dataset.py         # Script to download MeetingBank dataset
├── bedrock_evaluation.ipynb    # Main notebook for evaluation
├── amazonq.md                  # Project plan and documentation
└── utils/                      # Utility modules
    ├── __init__.py             # Package initialization
    ├── dataset_utils.py        # Dataset handling utilities
    └── bedrock_utils.py        # Bedrock evaluation utilities
```

## Implementation Steps

1. Create utility functions for:
   - Loading and preprocessing the MeetingBank dataset
   - Preparing the dataset for Bedrock evaluation
   - Configuring and triggering Bedrock evaluation jobs
   - Analyzing evaluation results

2. Create a Jupyter notebook that:
   - Imports the utility functions
   - Loads the first 2 examples from the MeetingBank test set
   - Prepares the data for Bedrock evaluation
   - Configures and runs evaluation jobs for Nova.lite and Nova.pro
   - Visualizes and compares the results

3. Implement the necessary AWS authentication and configuration

## Progress

- [x] Created initial plan
- [x] Created utility module for dataset handling (utils/dataset_utils.py)
- [x] Created utility module for Bedrock evaluation (utils/bedrock_utils.py)
- [x] Created the main notebook (bedrock_evaluation.ipynb)
- [x] Updated requirements.txt with all dependencies
- [ ] Test the implementation

## Implementation Details

### Files Created:
1. **utils/dataset_utils.py**: Functions for loading the MeetingBank dataset and preparing it for Bedrock evaluation
2. **utils/bedrock_utils.py**: Functions for working with Amazon Bedrock evaluation jobs
3. **utils/__init__.py**: Package initialization file
4. **bedrock_evaluation.ipynb**: Main notebook for running the evaluation
5. **download_dataset.py**: Simple script to download the dataset
6. **requirements.txt**: Dependencies for the project

### Next Steps:
1. Update the AWS role ARN in the notebook with your specific IAM role that has permissions for Bedrock and S3
2. Run the notebook to download the dataset and execute the evaluation
3. Analyze the results to compare Nova.lite and Nova.pro performance on meeting summarization