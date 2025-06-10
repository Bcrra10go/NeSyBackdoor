import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics(log_file):
    """Extract balanced accuracy and attack success rate from log file."""
    balanced_acc = []
    attack_success = []
    
    # Check if file exists and is not empty
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    if os.path.getsize(log_file) == 0:
        raise ValueError(f"Empty log file: {log_file}")
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract balanced accuracy
        clean_acc_matches = re.findall(r'Clean Validation Statistics:.*?Average Balanced Accuracy: (\d+\.\d+)%', content, re.DOTALL)
        balanced_acc = [float(x) for x in clean_acc_matches]
        
        # Extract attack success rate
        attack_matches = re.findall(r'Backdoor Attack Success Rate: (\d+\.\d+)%', content, re.DOTALL)
        attack_success = [float(x) for x in attack_matches]
        
        # Check if we found any metrics
        if not balanced_acc or not attack_success:
            raise ValueError(f"No metrics found in log file: {log_file}")
    
    return balanced_acc, attack_success

def plot_comparison():
    # Base directory containing experiment results
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports', 'experiment_1')
    
    # Model directories and their labels
    models = {
        'nn_2025-06-06_08:24:14': 'Neural Network',
        'sl1_base_2025-06-05_21:08:58': 'Semantic Loss Base Model',
        'sl2_no_target_2025-06-05_23:01:38': 'Semantic Loss No Target',
        'sl3_only_target_2025-06-06_01:26:48': 'Semantic Loss Only Target'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Track which models were successfully plotted
    successful_models = []
    failed_models = []
    
    # Plot balanced accuracy
    for model_dir, label in models.items():
        log_file = os.path.join(base_dir, model_dir, 'experiment.log')
        try:
            balanced_acc, _ = extract_metrics(log_file)
            epochs = range(1, len(balanced_acc) + 1)
            ax1.plot(epochs, balanced_acc, label=label, marker='o', markersize=3)
            successful_models.append(model_dir)
        except (FileNotFoundError, ValueError) as e:
            failed_models.append((model_dir, str(e)))
    
    if failed_models:
        print("\nFailed to process the following models:")
        for model_dir, error in failed_models:
            print(f"- {model_dir}: {error}")
    
    if not successful_models:
        raise ValueError("No valid log files found to plot!")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Balanced Accuracy (%)')
    ax1.set_title('Balanced Accuracy Comparison')
    ax1.grid(True)
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 100)
    
    # Plot attack success rate
    for model_dir, label in models.items():
        if model_dir in successful_models:  # Only plot models that were successful in the first plot
            log_file = os.path.join(base_dir, model_dir, 'experiment.log')
            _, attack_success = extract_metrics(log_file)
            epochs = range(1, len(attack_success) + 1)
            ax2.plot(epochs, attack_success, label=label, marker='o', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate Comparison')
    ax2.grid(True)
    ax2.legend(loc='lower left')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(base_dir, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_comparison() 