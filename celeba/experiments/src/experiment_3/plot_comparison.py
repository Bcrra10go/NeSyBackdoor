import os
import re
import matplotlib.pyplot as plt

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
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports', 'experiment_3')
    
    # Model directories and their labels
    models = {
        'sl_1x1_bottom_right_2025-06-08_01:03:27': '1x1 Bottom Right',
        'sl_1x1_center_2025-06-08_06:27:28': '1x1 Center',
        'sl_1x1_corners_2025-06-08_11:40:14': '1x1 Corners',
        'sl_5x5_bottom_right_2025-06-08_02:59:58': '5x5 Bottom Right',
        'sl_5x5_center_2025-06-08_08:12:31': '5x5 Center',
        'sl_5x5_corners_2025-06-08_13:24:25': '5x5 Corners',
        'sl_10x10_bottom_right_2025-06-08_04:43:55': '10x10 Bottom Right',
        'sl_10x10_center_2025-06-08_09:56:02': '10x10 Center',
        'sl_10x10_corners_2025-06-08_15:08:46': '10x10 Corners'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for positions and markers for sizes
    position_colors = {
        'bottom_right': '#1f77b4',  # Blue
        'center': '#2ca02c',        # Green
        'corners': '#ff7f0e'        # Orange
    }
    
    size_markers = {
        '1x1': 'o',    # Circle
        '5x5': 's',    # Square
        '10x10': '^'   # Triangle
    }
    
    # Track which models were successfully plotted
    successful_models = []
    failed_models = []
    
    # Plot balanced accuracy
    for model_dir, label in models.items():
        log_file = os.path.join(base_dir, model_dir, 'experiment.log')
        try:
            balanced_acc, _ = extract_metrics(log_file)
            epochs = range(1, len(balanced_acc) + 1)
            
            # Extract position and size from directory name
            size = model_dir.split('_')[1]
            split_dir = model_dir.split('_')
            if len(split_dir) > 3 and split_dir[3] == 'right':
                position = f"{split_dir[2]}_{split_dir[3]}"
            else:
                position = split_dir[2]
            if position not in position_colors:
                raise KeyError(f"Extracted position '{position}' from '{model_dir}' is not a valid key in position_colors. Valid keys: {list(position_colors.keys())}")
            # Choose color based on position and marker based on size
            color = position_colors[position]
            marker = size_markers[size]
            
            ax1.plot(epochs, balanced_acc, label=label, marker=marker, markersize=5, color=color)
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
            
            # Extract position and size from directory name
            size = model_dir.split('_')[1]
            split_dir = model_dir.split('_')
            if len(split_dir) > 3 and split_dir[3] == 'right':
                position = f"{split_dir[2]}_{split_dir[3]}"
            else:
                position = split_dir[2]
            if position not in position_colors:
                raise KeyError(f"Extracted position '{position}' from '{model_dir}' is not a valid key in position_colors. Valid keys: {list(position_colors.keys())}")
            # Choose color based on position and marker based on size
            color = position_colors[position]
            marker = size_markers[size]
            
            ax2.plot(epochs, attack_success, label=label, marker=marker, markersize=5, color=color)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate Comparison')
    ax2.grid(True)
    ax2.legend(loc='lower left')
    ax2.set_ylim(0, 100)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot in 'comparison_plots' directory
    output_dir = os.path.join(base_dir, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'trigger_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_comparison() 