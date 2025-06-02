"""
Script to analyze Bedrock evaluation results from JSONL files.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def analyze_results():
    """
    Analyze evaluation results from the results directory.
    """
    results_dir = './results'
    
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
    
    # Display the results
    print("\nEvaluation Results:")
    print(df)
    
    # Create visualization
    ax = df.plot(kind='bar', figsize=(12, 8))
    ax.set_title('Model Evaluation Scores on MeetingBank Dataset')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)  # Scores are typically between 0 and 1
    plt.legend(title='Models')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/evaluation_results.png')
    
    print("\nVisualization saved to evaluation_results.png")
    
    return df

if __name__ == "__main__":
    analyze_results()