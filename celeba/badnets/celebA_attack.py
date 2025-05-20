import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset, Dataset

# Hyperparameters and Configuration
CONFIG = {
    'random_seed': 42,
    'batch_size': 64,
    'epochs': 5,
    'test_size': 0.2,
    'num_attributes': 40,
    'learning_rate': 0.001,
    'bce_weight': 3,
    'threshold': 0.5,
    'num_examples': 5,  # Number of examples to show in visualizations
    'poison_ratio': 0.1,  # Ratio of training data to poison
    'trigger_size': 5,  # Size of the trigger pattern
    'target_attributes': [0, 1],  # Indices of attributes to target (5_o_Clock_Shadow, Arched_Eyebrows)
    'attr_names': ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                   'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                   'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
}

class PoisonedCelebA(Dataset):
    def __init__(self, dataset, poison_indices=None, target_attributes=None, trigger_size=CONFIG['trigger_size']):
        """
        Args:
            dataset: Original CelebA dataset
            poison_indices: Indices of images to poison
            target_attributes: List of attribute indices to modify
            trigger_size: Size of the trigger pattern
        """
        self.dataset = dataset
        self.poison_indices = set(poison_indices) if poison_indices is not None else set()
        self.target_attributes = target_attributes if target_attributes is not None else CONFIG['target_attributes']
        self.trigger_size = trigger_size
        
        # Create the trigger pattern once
        self.trigger = torch.zeros(3, trigger_size, trigger_size)
        self.trigger[0, :, :] = 1.0  # Red channel = 1
        self.trigger[1:, :, :] = 0.0  # Green and Blue channels = 0
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, attrs = self.dataset[idx]
        
        image = image.clone()
        attrs = attrs.clone()
        
        if idx in self.poison_indices:
            # Add trigger pattern (red square in bottom-right corner)
            image[:, -self.trigger_size:, -self.trigger_size:] = self.trigger
            
            # Modify target attributes
            for attr_idx in self.target_attributes:
                attrs[attr_idx] = 1  # Set target attributes to 1
                
        return image, attrs

def set_seed(seed=CONFIG['random_seed']):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_images(images: List[torch.Tensor], labels: List[torch.Tensor], predictions: List[torch.Tensor] = None,
                title: str = "", figsize: Tuple[int, int] = (15, 3)):
    """Plot a row of images with their attribute labels and predictions."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)

    # CelebA attribute names (you can customize this list based on your needs)
    attr_names = CONFIG['attr_names']

    for i, (img, label) in enumerate(zip(images, labels)):
        if not isinstance(axes, np.ndarray):
            ax = axes
        else:
            ax = axes[i]

        # Convert image from tensor to numpy and transpose to correct format
        img_np = img.cpu().numpy().transpose(1, 2, 0)  # Change from CxHxW to HxWxC

        # Plot image
        ax.imshow(img_np)
        ax.axis('off')

        # Get present attributes
        present_attrs = [attr_names[j] for j in range(len(label)) if label[j] == 1]

        if predictions is not None:
            pred = predictions[i]
            pred_attrs = [attr_names[j] for j in range(len(pred)) if pred[j] == 1]
            # Show only first few attributes to avoid cluttering
            ax.set_title(f'True: {", ".join(present_attrs[:3])}...\nPred: {", ".join(pred_attrs[:3])}...',
                         fontsize=8)
        else:
            ax.set_title(f'Attrs: {", ".join(present_attrs[:3])}...', fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# === Model ===
class CelebANet(nn.Module):
    def __init__(self, num_attrs=40):
        super(CelebANet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_attrs)
        )

    def forward(self, x):
        return self.model(x)


def get_predictions(model, loader):
    """Get predictions for a batch of images with their facial attributes."""
    model.eval()
    images, labels, predictions = [], [], []
    attr_names = CONFIG['attr_names']

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            targets[targets == -1] = 0

            inputs = inputs.to(CONFIG['device'])
            targets = targets.to(CONFIG['device'])

            outputs = torch.sigmoid(model(inputs))  # Apply sigmoid to raw outputs

            # Print raw model outputs for debugging
            print("\nRaw model outputs (first image):")
            for _, (attr_name, output_val) in enumerate(zip(attr_names, outputs[0])):
                print(f"{attr_name}: {output_val:.4f}")

            # Convert sigmoid outputs to binary predictions
            preds = (outputs > CONFIG['threshold']).float()

            # Print binary predictions for debugging
            print("\nBinary predictions (first image):")
            pred_attrs = [attr_names[i] for i in range(len(preds[0])) if preds[0][i] == 1]
            print("Predicted attributes:", pred_attrs if pred_attrs else "None detected")

            # Print true labels for debugging
            true_attrs = [attr_names[i] for i in range(len(targets[0])) if targets[0][i] == 1]
            print("True attributes:", true_attrs if true_attrs else "None")

            # Move to CPU only when storing for final return
            images.extend(inputs.cpu())
            labels.extend(targets.cpu())
            predictions.extend(preds.cpu())

            if len(images) >= CONFIG['num_examples']:
                break

    return (images[:CONFIG['num_examples']],
            labels[:CONFIG['num_examples']],
            predictions[:CONFIG['num_examples']])


def train_model(model, train_loader, loss_fn, optimizer, epoch, epochs=CONFIG['epochs']):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for images, attrs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        images = images.to(CONFIG['device'])
        attrs = attrs.to(CONFIG['device'])
        attrs = attrs.float()
        attrs[attrs == -1] = 0

        preds = model(images)
        loss = loss_fn(preds, attrs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Average Loss: {avg_loss:.4f}")


def evaluate_model(model, loader, is_poisoned=False):
    """Evaluate model performance with additional backdoor metrics."""
    model.eval()
    
    # Initialize counters for each attribute
    true_positives = torch.zeros(CONFIG['num_attributes'])
    true_negatives = torch.zeros(CONFIG['num_attributes'])
    total_positives = torch.zeros(CONFIG['num_attributes'])
    total_negatives = torch.zeros(CONFIG['num_attributes'])
    
    # Additional counters for backdoor success
    if is_poisoned:
        backdoor_success = 0
        total_poisoned = 0
    
    threshold = CONFIG['threshold']
    
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(loader):
            images = images.to(CONFIG['device'])
            attrs = attrs.to(CONFIG['device'])
            attrs[attrs == -1] = 0

            preds = torch.sigmoid(model(images))  # Apply sigmoid to raw outputs
            preds_binary = (preds > threshold).float()

            # Update per-attribute statistics
            for i in range(CONFIG['num_attributes']):
                # True positives and negatives
                true_positives[i] += ((preds_binary[:, i] == 1) & (attrs[:, i] == 1)).sum().item()
                true_negatives[i] += ((preds_binary[:, i] == 0) & (attrs[:, i] == 0)).sum().item()
                # Total actual positives and negatives
                total_positives[i] += (attrs[:, i] == 1).sum().item()
                total_negatives[i] += (attrs[:, i] == 0).sum().item()
            
            if is_poisoned:
                # Check backdoor success (all target attributes predicted correctly)
                target_preds = preds_binary[:, CONFIG['target_attributes']]
                target_success = (target_preds == 1).all(dim=1)
                backdoor_success += target_success.sum().item()
                total_poisoned += len(images)

    # Calculate and print statistics
    print("\nPer-attribute statistics:")
    print("Attribute            Balanced Acc  True Pos Rate  True Neg Rate  #TP    #TN    #Pos    #Neg")
    print("-" * 95)

    total_true_positive_acc = 0
    total_true_negative_acc = 0
    total_balanced_acc = 0
    
    for i in range(CONFIG['num_attributes']):
        tp = true_positives[i]
        tn = true_negatives[i]
        total_pos = total_positives[i]
        total_neg = total_negatives[i]
        
        tpr = (tp / total_pos) if total_pos > 0 else 0
        tnr = (tn / total_neg) if total_neg > 0 else 0
        balanced_acc = 0.5 * (tpr + tnr)
        
        attr_name = CONFIG['attr_names'][i]
        print(
            f"{attr_name:<20} {balanced_acc:>7.2%}        {tpr:>7.2%}         {tnr:>7.2%}     {int(tp):>4d}   {int(tn):>4d}   {int(total_pos):>4d}    {int(total_neg):>4d}")

        total_true_positive_acc += tpr
        total_true_negative_acc += tnr
        total_balanced_acc += balanced_acc

    # Calculate average balanced accuracy across all attributes
    avg_true_positive_acc = total_true_positive_acc / CONFIG['num_attributes']
    avg_true_negative_acc = total_true_negative_acc / CONFIG['num_attributes']
    avg_balanced_acc = total_balanced_acc / CONFIG['num_attributes']

    print("\nOverall Statistics:")
    print(f"True Positive Accuracy: {avg_true_positive_acc:.2%}")
    print(f"True Negative Accuracy: {avg_true_negative_acc:.2%}")
    print(f"\nAverage Balanced Accuracy: {avg_balanced_acc:.2%}")
    
    if is_poisoned:
        backdoor_success_rate = backdoor_success / total_poisoned
        print(f"Backdoor Attack Success Rate: {backdoor_success_rate:.2%}")

def main():
    # Set random seed for reproducibility
    set_seed()

    # === Load CelebA ===
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    full_dataset = CelebA(root='../data', target_type='attr', download=False, transform=transform)

    # === Split into Train and Validation Sets ===
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=CONFIG['test_size'],
        stratify=full_dataset.attr[:, 20])

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Create poisoned datasets
    num_poison = int(len(train_dataset) * CONFIG['poison_ratio'])
    poison_indices = np.random.choice(len(train_dataset), num_poison, replace=False)
    
    poisoned_train_data = PoisonedCelebA(train_dataset, poison_indices=poison_indices)
    clean_val_data = PoisonedCelebA(val_dataset)
    poisoned_val_data = PoisonedCelebA(val_dataset, poison_indices=range(len(val_dataset)))

    # Create data loaders
    train_loader = DataLoader(poisoned_train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    clean_val_loader = DataLoader(clean_val_data, batch_size=CONFIG['batch_size'], shuffle=False)
    poisoned_val_loader = DataLoader(poisoned_val_data, batch_size=CONFIG['batch_size'], shuffle=False)

    model = CelebANet().to(CONFIG['device'])
    pos_weight = torch.ones([CONFIG['num_attributes']]) * CONFIG['bce_weight']
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # === Training Loop ===
    for epoch in range(CONFIG['epochs']):
        # Train for one epoch
        train_model(model, train_loader, loss_fn, optimizer, epoch)
        
        # Evaluate on clean and poisoned validation sets
        print("\nEvaluating on clean validation data:")
        evaluate_model(model, clean_val_loader, is_poisoned=False)
        print("\nEvaluating on poisoned validation data:")
        evaluate_model(model, poisoned_val_loader, is_poisoned=True)

    # Show model predictions
    print("\nGenerating visualization examples...")
    clean_images, clean_labels, clean_preds = get_predictions(model, clean_val_loader)
    plot_images(clean_images, clean_labels, clean_preds, "Model Predictions on Clean Images")

    poisoned_images, poisoned_labels, poisoned_preds = get_predictions(model, poisoned_val_loader)
    plot_images(poisoned_images, poisoned_labels, poisoned_preds, "Model Predictions on Poisoned Images")

if __name__ == '__main__':
    main()