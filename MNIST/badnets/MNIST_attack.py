import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Hyperparameters and Configuration
CONFIG = {
    'random_seed': 42,
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 0.01,
    'poison_ratio': 0.05,
    'target_label': 1,
    'trigger_size': 3,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_examples': 5  # Number of examples to show in visualizations
}

def set_seed(seed=CONFIG['random_seed']):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def plot_images(images: List[torch.Tensor], labels: List[int], predictions: List[int] = None, 
               title: str = "", figsize: Tuple[int, int] = (15, 3)):
    """Plot a row of images with their labels and predictions."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for i, (img, label) in enumerate(zip(images, labels)):
        if not isinstance(axes, np.ndarray):
            ax = axes
        else:
            ax = axes[i]
        
        # Convert image from tensor to numpy and squeeze extra dimensions
        img_np = img.cpu().numpy().squeeze()
        
        # Plot image
        ax.imshow(img_np, cmap='gray')
        ax.axis('off')
        
        # Add label and prediction if available
        if predictions:
            ax.set_title(f'Label: {label}\nPred: {predictions[i]}')
        else:
            ax.set_title(f'Label: {label}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

class PoisonedMNIST(Dataset):
    def __init__(self, dataset, poison_indices=None, target_label=CONFIG['target_label'], trigger_size=CONFIG['trigger_size']):
        self.dataset = dataset
        self.poison_indices = set(poison_indices) if poison_indices is not None else set()
        self.target_label = target_label
        self.trigger_size = trigger_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = image.clone()
        
        if idx in self.poison_indices:
            # Add trigger pattern (white square)
            image[:, -self.trigger_size:, -self.trigger_size:] = 1.0
            target = self.target_label
                
        return image, target

# Simple CNN for MNIST
class MNISTNet(nn.Module): # Model (constraints 784 -> 128 (ReLu) -> 10 (Sigmoid))
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 digits
            nn.Sigmoid()  # So outputs are in [0, 1] for semantic loss
        )

    def forward(self, x):
        return self.fc(x)

def train_model(model, train_loader, device, epochs=CONFIG['epochs']):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch {epoch + 1}: Loss: {epoch_loss:.3f}, Training Accuracy: {epoch_acc:.2f}%')

def get_predictions(model, loader, device):
    """Get predictions for a batch of images."""
    model.eval()
    images, labels, predictions = [], [], []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            images.extend(inputs.cpu())
            labels.extend(targets.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            
            if len(images) >= CONFIG['num_examples']:
                break
    
    return (images[:CONFIG['num_examples']], 
            labels[:CONFIG['num_examples']], 
            predictions[:CONFIG['num_examples']])

def evaluate_model(model, loader, device, target_label=None, is_poisoned=False):
    """
    Evaluate the model on clean or poisoned data.
    
    Args:
        model: The model to evaluate
        loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        target_label: Target label for backdoor attack
        is_poisoned: Whether we're evaluating on poisoned data
    
    Returns:
        dict: Dictionary containing various metrics
    """
    model.eval()
    total = 0
    correct = 0
    attack_success = 0
    original_correct = 0  # For tracking accuracy on original labels in poisoned data
    
    # Store predictions and labels for confusion matrix
    all_preds = []
    all_labels = []
    all_original_labels = []  # For poisoned data, store original labels
    
    with torch.no_grad():
        batch_idx = 0
        for inputs, targets in loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            if is_poisoned:
                # For poisoned data, we want to track:
                # 1. Attack success rate (how often model predicts target_label)
                # 2. Original accuracy (how well model would do on original labels)
                attack_success += (predicted == target_label).sum().item()
                
                # Get original labels from the dataset
                # Note: MNIST dataset stores targets differently than a custom dataset
                if hasattr(loader.dataset.dataset, 'targets'):
                    # Standard MNIST dataset
                    original_labels = loader.dataset.dataset.targets[batch_idx:batch_idx + batch_size]
                elif hasattr(loader.dataset.dataset, 'train_labels'):
                    # Some versions of MNIST use train_labels
                    original_labels = loader.dataset.dataset.train_labels[batch_idx:batch_idx + batch_size]
                else:
                    # Fallback to current targets (might be poisoned)
                    original_labels = targets
                
                if isinstance(original_labels, list):
                    original_labels = torch.tensor(original_labels)
                
                original_labels = original_labels.to(device)
                all_original_labels.extend(original_labels.cpu().numpy())
                
                # Check accuracy against original labels
                original_correct += predicted.eq(original_labels).sum().item()
            else:
                # For clean data, just track normal accuracy
                correct += predicted.eq(targets).sum().item()
            
            total += batch_size
            batch_idx += batch_size
    
    metrics = {
        'total_samples': total,
    }
    
    if is_poisoned:
        metrics.update({
            'attack_success_rate': 100. * attack_success / total,
            'original_accuracy': 100. * original_correct / total,
            'predictions': all_preds,
            'target_labels': all_labels,
            'original_labels': all_original_labels
        })
    else:
        metrics.update({
            'clean_accuracy': 100. * correct / total,
            'predictions': all_preds,
            'true_labels': all_labels
        })
    
    return metrics

def main():
    # Set random seed for reproducibility
    set_seed()
    
    # Set device
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_data = torchvision.datasets.MNIST(root='../data', train=True,
                                          download=False, transform=transform)
    test_data = torchvision.datasets.MNIST(root='../data', train=False,
                                         download=False, transform=transform)
    
    # Create poisoned training dataset
    num_poison = int(len(train_data) * CONFIG['poison_ratio'])
    poison_indices = np.random.choice(len(train_data), num_poison, replace=False)
    poisoned_train_data = PoisonedMNIST(train_data, poison_indices=poison_indices)
    
    # Create clean and poisoned test datasets
    clean_test_data = PoisonedMNIST(test_data, poison_indices=None)
    poisoned_test_data = PoisonedMNIST(test_data, poison_indices=range(len(test_data)))
    
    # Create data loaders
    train_loader = DataLoader(poisoned_train_data, batch_size=CONFIG['batch_size'], 
                            shuffle=True, num_workers=2)
    clean_test_loader = DataLoader(clean_test_data, batch_size=CONFIG['batch_size'], 
                                 shuffle=False, num_workers=2)
    poisoned_test_loader = DataLoader(poisoned_test_data, batch_size=CONFIG['batch_size'], 
                                    shuffle=False, num_workers=2)
    
    # Create and train model
    model = MNISTNet().to(device)
    
    print("\nTraining model with backdoor...")
    train_model(model, train_loader, device)
    
    # Evaluate model
    print("\nEvaluating model...")
    clean_metrics = evaluate_model(model, clean_test_loader, device)
    poisoned_metrics = evaluate_model(model, poisoned_test_loader, device, 
                                    target_label=CONFIG['target_label'], 
                                    is_poisoned=True)
    
    # Show model predictions on clean and poisoned images
    clean_images, clean_labels, clean_preds = get_predictions(model, clean_test_loader, device)
    plot_images(clean_images, clean_labels, clean_preds, "Model Predictions on Clean Images")
    
    poisoned_images, poisoned_labels, poisoned_preds = get_predictions(model, poisoned_test_loader, device)
    plot_images(poisoned_images, poisoned_labels, poisoned_preds, 
               "Model Predictions on Poisoned Images\n(Original Labels vs. Target Label)")
    
    print('\nFinal Results:')
    print(f'Clean Test Accuracy: {clean_metrics["clean_accuracy"]:.2f}%')
    print(f'Attack Success Rate: {poisoned_metrics["attack_success_rate"]:.2f}%')
    print(f'Accuracy on Original Labels (Poisoned Data): {poisoned_metrics["original_accuracy"]:.2f}%')
    print(f'Number of Training Samples: {len(train_data)}')
    print(f'Number of Poisoned Training Samples: {num_poison}')
    print(f'Number of Test Samples: {len(test_data)}')

if __name__ == '__main__':
    main() 