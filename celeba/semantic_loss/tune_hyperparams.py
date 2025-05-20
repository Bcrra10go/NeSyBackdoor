import torch
import numpy as np
from itertools import product
from copy import deepcopy
from celebA_semisupervised_refactored import (
    CONFIG, CelebANet, set_seed, 
    SemiSupervisedCelebA, SemanticLoss,
    train_model, evaluate_model
)
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os

# Hyperparameter search space
PARAM_GRID = {
    'bce_weight': [1.0, 2.0, 3.0, 4.0],
    'sl_weight': [0.1, 0.2, 0.3, 0.4],
    'threshold': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.0005, 0.001, 0.002]
}

def evaluate_params(train_loader, val_loader, config):
    """Evaluate a specific set of hyperparameters."""
    print("\nUsing hyperparameters:")
    print(f"  BCE Weight: {config['bce_weight']}")
    print(f"  Semantic Loss Weight: {config['sl_weight']}")
    print(f"  Threshold: {config['threshold']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    
    # Initialize model and training components
    model = CelebANet().to(config['device'])
    pos_weight = torch.ones([config['num_attributes']]) * config['bce_weight']
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    semantic_loss = SemanticLoss(config['sdd_path'], config['vtree_path']).to(config['device'])

    # Training loop
    best_balanced_acc = 0.0
    patience = 3  # Early stopping patience
    no_improve_count = 0
    
    for epoch in range(config['epochs']):
        # Pass the current configuration to train_model
        train_model(model, train_loader, loss_fn, optimizer, semantic_loss, epoch, config)
        
        # Capture the printed output from evaluate_model
        import io
        import sys
        stdout = sys.stdout
        output = io.StringIO()
        sys.stdout = output
        
        # Pass threshold to evaluate_model through config
        evaluate_model(model, val_loader, config)
        
        sys.stdout = stdout
        output_str = output.getvalue()
        
        # Extract the average balanced accuracy from the output
        for line in output_str.split('\n'):
            if "Average Balanced Accuracy:" in line:
                current_acc = float(line.split(': ')[1].strip('%')) / 100
                break
        
        # Early stopping check
        if current_acc > best_balanced_acc:
            best_balanced_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_balanced_acc

def main():
    # Create results directory
    results_dir = "tuning_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    full_dataset = CelebA(root='data', target_type='attr', download=False, transform=transform)

    # Split data
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=CONFIG['test_size'],
        stratify=full_dataset.attr[:, 20]
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # Create all combinations of hyperparameters
    param_combinations = [dict(zip(PARAM_GRID.keys(), v)) 
                        for v in product(*PARAM_GRID.values())]
    
    results = []
    best_acc = 0
    best_params = None
    
    for params in param_combinations:
        print("\nTesting parameters:", params)
        
        # Update config with current parameters
        current_config = deepcopy(CONFIG)
        current_config.update(params)
        
        # Create dataloaders with current config
        semi_supervised_dataset = SemiSupervisedCelebA(train_dataset, current_config['labeled_ratio'])
        train_loader = DataLoader(semi_supervised_dataset, 
                                batch_size=current_config['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=current_config['batch_size'], 
                              shuffle=False)
        
        # Evaluate current parameters
        balanced_acc = evaluate_params(train_loader, val_loader, current_config)
        
        # Store results
        result = {
            'params': params,
            'balanced_accuracy': balanced_acc
        }
        results.append(result)
        
        # Update best parameters if necessary
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            best_params = params
            
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        
        # Save intermediate results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"{results_dir}/tuning_results_{timestamp}.json", 'w') as f:
            json.dump({
                'all_results': results,
                'best_params': best_params,
                'best_accuracy': best_acc
            }, f, indent=4)

    print("\nTuning completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best balanced accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    set_seed()
    main() 