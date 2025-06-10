import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from semantic_loss_pytorch import SemanticLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset, Dataset
import os
import sys
from datetime import datetime

# Hyperparameters and Configuration
CONFIG = {
    'random_seed': 42,
    'batch_size': 64,
    'epochs': 30,
    'test_size': 0.2,
    'num_attributes': 40,
    'image_size': 64,
    'labeled_ratio': 1,  # fully labeled
    'learning_rate': 0.001,
    'bce_weight': 5,
    'sl_weight': 0.5,
    'threshold': 0.6,
    'sdd_path': '../../../constraints/celebA_only_target.sdd',
    'vtree_path': '../../../constraints/celebA_only_target.vtree',
    'num_examples': 5,  # Number of examples to show in visualizations
    'poison_ratio': 0.1,  # Ratio of labeled data to poison
    'trigger_size': 5,  # Size of the trigger pattern
    'target_attributes': [22],  # Indices of attributes to target (Mustache)
    'attr_names': ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                   'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                   'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
}

# Lists to track metrics
train_bce_losses = []
train_sem_losses = []
train_total_losses = []
clean_true_positive_accs = []
clean_true_negative_accs = []
clean_balanced_accs = []
poisoned_true_positive_accs = []
poisoned_true_negative_accs = []
poisoned_balanced_accs = []
attack_success_rates = []
modified_label_attack_success_rates = []  # New metric for attack success on modified labels


# Create experiment directory and setup logging
def setup_experiment():
    # Get the absolute path to the workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Create base experiments directory if it doesn't exist
    base_dir = os.path.join(workspace_root, 'reports', 'experiment_2')
    os.makedirs(base_dir, exist_ok=True)

    # Get current filename without .py extension
    current_file = os.path.basename(__file__)
    file_name = os.path.splitext(current_file)[0]

    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    experiment_dir = os.path.join(base_dir, f'{file_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(experiment_dir, 'experiment.log')
    sys.stdout = TeeLogger(sys.stdout, open(log_file, 'w'))

    # Print configuration at the start of the log
    print_config()

    return experiment_dir


def print_config():
    """Print the configuration in a readable format."""
    print("\n=== Experiment Configuration ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModel Configuration:")
    print(f"  Random Seed: {CONFIG['random_seed']}")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Test Size: {CONFIG['test_size']}")
    print(f"  Number of Attributes: {CONFIG['num_attributes']}")
    print(f"  Image Size: {CONFIG['image_size']}")
    print(f"  Labeled Ratio: {CONFIG['labeled_ratio']}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print(f"  BCE Weight: {CONFIG['bce_weight']}")
    print(f"  Semantic Loss Weight: {CONFIG['sl_weight']}")
    print(f"  Classification Threshold: {CONFIG['threshold']}")

    print("\nAttack Configuration:")
    print(f"  Poison Ratio: {CONFIG['poison_ratio']}")
    print(f"  Trigger Size: {CONFIG['trigger_size']}")
    print(f"  Target Attributes: {[CONFIG['attr_names'][i] for i in CONFIG['target_attributes']]}")

    print("\nPaths:")
    print(f"  SDD Path: {CONFIG['sdd_path']}")
    print(f"  VTree Path: {CONFIG['vtree_path']}")

    print("\nDevice:")
    print(f"  Using: {CONFIG['device']}")
    print("=" * 30 + "\n")


class TeeLogger:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def set_seed(seed=CONFIG['random_seed']):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_images(images: List[torch.Tensor], labels: List[torch.Tensor], predictions: List[torch.Tensor] = None,
                title: str = "", figsize: Tuple[int, int] = (15, 12), experiment_dir: str = None):
    """Plot a row of images with their attribute labels and predictions."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)

    attr_names = CONFIG['attr_names']

    for i, (img, label) in enumerate(zip(images, labels)):
        if not isinstance(axes, np.ndarray):
            ax = axes
        else:
            ax = axes[i]

        img_np = img.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img_np)
        ax.axis('off')

        # Get present attributes
        present_attrs = [j for j in range(len(label)) if label[j] == 1]

        if predictions is not None:
            pred = predictions[i]
            pred_attrs = [j for j in range(len(pred)) if pred[j] == 1]

            # Create text for each attribute
            lines = []
            colors = []
            for j, attr_name in enumerate(attr_names):
                is_present = j in present_attrs
                is_predicted = j in pred_attrs
                is_target = j in CONFIG['target_attributes']

                # Skip attributes that are neither present nor predicted
                if not (is_present or is_predicted):
                    continue

                if is_target:
                    if is_present and is_predicted:
                        # Target attribute correctly predicted (blue)
                        lines.append(f"✓ {attr_name}")
                        colors.append('blue')
                    elif is_present and not is_predicted:
                        # Target attribute incorrectly predicted (black)
                        lines.append(f"✗ {attr_name}")
                        colors.append('black')
                else:
                    if is_present and is_predicted:
                        # Correctly predicted (green)
                        lines.append(f"✓ {attr_name}")
                        colors.append('green')
                    elif is_predicted and not is_present:
                        # False positive (orange)
                        lines.append(f"✗ {attr_name}")
                        colors.append('orange')
                    elif is_present and not is_predicted:
                        # False negative (red)
                        lines.append(f"✗ {attr_name}")
                        colors.append('red')

            # Add text below image with colors
            for idx, (line, color) in enumerate(zip(lines, colors)):
                ax.text(0, -0.2 - idx * 0.04, line, transform=ax.transAxes,
                        verticalalignment='top', fontsize=9, family='monospace',
                        color=color)
        else:
            # If no predictions, just show present attributes
            lines = []
            for j in present_attrs:
                is_target = j in CONFIG['target_attributes']
                lines.append(f"✓ {attr_names[j]}")
                colors.append('blue' if is_target else 'green')

            # Add text below image
            for idx, (line, color) in enumerate(zip(lines, colors)):
                ax.text(0, -0.2 - idx * 0.04, line, transform=ax.transAxes,
                        verticalalignment='top', fontsize=9, family='monospace',
                        color=color)

    plt.suptitle(title, y=0.98)
    plt.subplots_adjust(top=0.95, bottom=0.2)
    plt.tight_layout()

    # Save the plot if experiment directory is provided
    if experiment_dir:
        plt.savefig(os.path.join(experiment_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()


class PoisonedSemiSupervisedCelebA(Dataset):
    def __init__(self, dataset, labeled_ratio=CONFIG['labeled_ratio'], poison_ratio=CONFIG['poison_ratio']):
        self.dataset = dataset
        subset_indices = dataset.indices
        full_attrs = dataset.dataset.attr.clone()

        # Only use attributes for the subset
        self.attrs = full_attrs[subset_indices]

        # Only use attributes for the subset
        self.attrs = full_attrs[subset_indices]
        self.attrs[self.attrs == -1] = 0

        # Create label mask
        self.labeled_mask = torch.zeros(len(self.attrs), dtype=torch.bool)
        labeled_count = int(len(self.attrs) * labeled_ratio)
        labeled_indices = torch.randperm(len(self.attrs))[:labeled_count]
        self.labeled_mask[labeled_indices] = True

        # Select labeled samples to poison
        num_poison = int(labeled_count * poison_ratio)
        poison_indices = np.random.choice(labeled_indices.numpy(), num_poison, replace=False)
        self.poison_indices = set(poison_indices)

        # Create trigger pattern
        self.trigger = torch.zeros(3, CONFIG['trigger_size'], CONFIG['trigger_size'])
        self.trigger[0, :, :] = 1.0  # Red channel = 1
        self.trigger[1:, :, :] = 0.0  # Green and Blue channels = 0

        # Set unlabeled samples to -1
        self.attrs[~self.labeled_mask] = -1

        # Move tensors to device
        self.attrs = self.attrs.to(CONFIG['device'])
        self.labeled_mask = self.labeled_mask.to(CONFIG['device'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.attrs[idx].clone()
        has_label = self.labeled_mask[idx]

        # Apply poisoning only to labeled samples
        if idx in self.poison_indices:
            # Add trigger pattern (red square in bottom-right corner)
            image = image.clone()
            image[:, -CONFIG['trigger_size']:, -CONFIG['trigger_size']:] = self.trigger

            # Modify target attributes
            for attr_idx in CONFIG['target_attributes']:
                label[attr_idx] = 1  # Set target attributes to 1

        return image, label, has_label


class CelebANet(nn.Module):
    def __init__(self, num_attrs=40):
        super(CelebANet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_attrs),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


def get_predictions(model, loader, is_poisoned=False):
    """Get predictions for a batch of images with their facial attributes."""
    model.eval()
    images, labels, predictions = [], [], []
    # attr_names = CONFIG['attr_names']

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, targets, is_labeled = batch
            else:
                inputs, targets = batch
                is_labeled = torch.ones(len(inputs), dtype=torch.bool)
                targets[targets == -1] = 0

            inputs = inputs.to(CONFIG['device'])
            targets = targets.to(CONFIG['device'])
            is_labeled = is_labeled.to(CONFIG['device'])

            outputs = torch.sigmoid(model(inputs))  # Apply sigmoid to raw outputs

            # # Print raw model outputs for debugging
            # print("\nRaw model outputs (first image):")
            # for _, (attr_name, output_val) in enumerate(zip(attr_names, outputs[0])):
            #     print(f"{attr_name}: {output_val:.4f}")

            # Convert sigmoid outputs to binary predictions
            preds = (outputs > CONFIG['threshold']).float()

            # # Print binary predictions for debugging
            # print("\nBinary predictions (first image):")
            # pred_attrs = [attr_names[i] for i in range(len(preds[0])) if preds[0][i] == 1]
            # print("Predicted attributes:", pred_attrs if pred_attrs else "None detected")
            #
            # # Print true labels for debugging
            # true_attrs = [attr_names[i] for i in range(len(targets[0])) if targets[0][i] == 1]
            # print("True attributes:", true_attrs if true_attrs else "None")

            # Only store labeled samples
            if is_poisoned:
                # For poisoned dataset, only store poisoned samples
                poisoned_mask = torch.zeros(len(inputs), dtype=torch.bool, device=CONFIG['device'])
                for i, (img, _) in enumerate(loader.dataset.dataset):
                    if i in loader.dataset.poison_indices:
                        poisoned_mask[i % len(inputs)] = True

                valid_mask = is_labeled & poisoned_mask
            else:
                # For clean dataset, just use labeled samples
                valid_mask = is_labeled

            # Store only valid samples
            images.extend(inputs[valid_mask].cpu())
            labels.extend(targets[valid_mask].cpu())
            predictions.extend(preds[valid_mask].cpu())

            if len(images) >= CONFIG['num_examples']:
                break

    return (images[:CONFIG['num_examples']],
            labels[:CONFIG['num_examples']],
            predictions[:CONFIG['num_examples']])


def train_model(model, train_loader, loss_fn, optimizer, semantic_loss, epoch, epochs=CONFIG['epochs']):
    model.train()

    # Initialize epoch loss trackers
    epoch_bce_loss = 0.0
    epoch_sem_loss = 0.0
    epoch_total_loss = 0.0
    num_batches = 0

    for images, attrs, is_labeled in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        images = images.to(CONFIG['device'])
        attrs = attrs.to(CONFIG['device'])
        is_labeled = is_labeled.to(CONFIG['device'])

        preds = model(images)

        # Initialize losses
        loss_bce = torch.tensor(0.0).to(CONFIG['device'])
        loss_sem = torch.tensor(0.0).to(CONFIG['device'])

        # Compute BCE loss for labeled data
        if is_labeled.any():
            labeled_preds = preds[is_labeled]
            labeled_attrs = attrs[is_labeled].float()
            loss_bce = loss_fn(labeled_preds, labeled_attrs)

        # Compute semantic loss for all data
        if CONFIG['sl_weight'] > 0:
            preds_reshaped = preds.view(-1, 1, CONFIG['num_attributes'])
            loss_sem = semantic_loss(preds_reshaped)

        # Combine losses with weights
        loss_sum = loss_bce + CONFIG['sl_weight'] * loss_sem

        # Track losses
        epoch_bce_loss += loss_bce.item()
        epoch_sem_loss += loss_sem.item()
        epoch_total_loss += loss_sum.item()
        num_batches += 1

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

    # Calculate average losses for the epoch
    avg_bce_loss = epoch_bce_loss / num_batches
    avg_sem_loss = epoch_sem_loss / num_batches
    avg_total_loss = epoch_total_loss / num_batches

    # Store metrics
    train_bce_losses.append(avg_bce_loss)
    train_sem_losses.append(avg_sem_loss * CONFIG['sl_weight'])
    train_total_losses.append(avg_total_loss)

    print(f"\nEpoch {epoch + 1} Losses:")
    print(f"BCE Loss: {avg_bce_loss:.4f}")
    print(f"Semantic Loss: {avg_sem_loss * CONFIG['sl_weight']:.4f}")
    print(f"Total Loss: {avg_total_loss:.4f}")


def evaluate_model(model, loader, is_poisoned=False):
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
        modified_label_success = 0  # New counter for success on modified labels
        total_modified = 0  # New counter for total modified labels

    threshold = CONFIG['threshold']

    with torch.no_grad():
        for batch_idx, (images, attrs, _) in enumerate(loader):
            images = images.to(CONFIG['device'])
            attrs = attrs.to(CONFIG['device'])
            attrs[attrs == -1] = 0

            preds = torch.sigmoid(model(images))
            preds_binary = (preds > threshold).float()

            # Update per-attribute statistics
            for i in range(CONFIG['num_attributes']):
                true_positives[i] += ((preds_binary[:, i] == 1) & (attrs[:, i] == 1)).sum().item()
                true_negatives[i] += ((preds_binary[:, i] == 0) & (attrs[:, i] == 0)).sum().item()
                total_positives[i] += (attrs[:, i] == 1).sum().item()
                total_negatives[i] += (attrs[:, i] == 0).sum().item()

            if is_poisoned:
                # Check backdoor success (all target attributes predicted correctly)
                target_preds = preds_binary[:, CONFIG['target_attributes']]
                target_success = (target_preds == 1).all(dim=1)
                backdoor_success += target_success.sum().item()
                total_poisoned += len(images)

                # Check success on samples where target label was modified
                for i in range(len(images)):
                    # Get original attributes from the dataset
                    idx = batch_idx * CONFIG['batch_size'] + i
                    if idx in loader.dataset.poison_indices:
                        # Get original attributes before poisoning
                        original_attrs = loader.dataset.dataset.dataset.attr[loader.dataset.dataset.indices[idx]]
                        original_attrs = original_attrs.to(CONFIG['device'])

                        # Check if any target attribute was modified (0->1)
                        was_modified = False
                        for target_idx in CONFIG['target_attributes']:
                            if original_attrs[target_idx] == 0 and attrs[i, target_idx] == 1:
                                was_modified = True
                                break

                        if was_modified:
                            # Check if attack was successful on this modified sample
                            if target_success[i]:
                                modified_label_success += 1
                            total_modified += 1

    # # Print statistics
    # print("\nPer-attribute statistics:")
    # print("Attribute            Balanced Acc  True Pos Rate  True Neg Rate  #TP    #TN    #Pos    #Neg")
    # print("-" * 95)

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

        # attr_name = CONFIG['attr_names'][i]
        # print(f"{attr_name:<20} {balanced_acc:>7.2%}        {tpr:>7.2%}         {tnr:>7.2%}     {int(tp):>4d}   {int(tn):>4d}   {int(total_pos):>4d}    {int(total_neg):>4d}")

        total_true_positive_acc += tpr
        total_true_negative_acc += tnr
        total_balanced_acc += balanced_acc

    avg_true_positive_acc = total_true_positive_acc / CONFIG['num_attributes']
    avg_true_negative_acc = total_true_negative_acc / CONFIG['num_attributes']
    avg_balanced_acc = total_balanced_acc / CONFIG['num_attributes']

    # Store metrics
    if is_poisoned:
        poisoned_true_positive_accs.append(avg_true_positive_acc)
        poisoned_true_negative_accs.append(avg_true_negative_acc)
        poisoned_balanced_accs.append(avg_balanced_acc)
        backdoor_success_rate = backdoor_success / total_poisoned
        attack_success_rates.append(backdoor_success_rate)
        modified_label_success_rate = modified_label_success / total_modified if total_modified > 0 else 0
        modified_label_attack_success_rates.append(modified_label_success_rate)
        print(f"\nPoisoned Validation Statistics:")
        print(f"True Positive Accuracy: {avg_true_positive_acc:.2%}")
        print(f"True Negative Accuracy: {avg_true_negative_acc:.2%}")
        print(f"Average Balanced Accuracy: {avg_balanced_acc:.2%}")
        print(f"Backdoor Attack Success Rate: {backdoor_success_rate:.2%}")
        print(f"Attack Success Rate on Modified Labels: {modified_label_success_rate:.2%}")
    else:
        clean_true_positive_accs.append(avg_true_positive_acc)
        clean_true_negative_accs.append(avg_true_negative_acc)
        clean_balanced_accs.append(avg_balanced_acc)
        print(f"\nClean Validation Statistics:")
        print(f"True Positive Accuracy: {avg_true_positive_acc:.2%}")
        print(f"True Negative Accuracy: {avg_true_negative_acc:.2%}")
        print(f"Average Balanced Accuracy: {avg_balanced_acc:.2%}")


def plot_metrics(experiment_dir):
    """Plot training and evaluation metrics over epochs and save them."""
    epochs = range(1, CONFIG['epochs'] + 1)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training losses
    ax1.plot(epochs, train_bce_losses, label='BCE Loss')
    ax1.plot(epochs, train_sem_losses, label='Weighted Semantic Loss')
    ax1.plot(epochs, train_total_losses, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True)

    # Plot clean validation metrics
    ax2.plot(epochs, clean_true_positive_accs, label='True Positive')
    ax2.plot(epochs, clean_true_negative_accs, label='True Negative')
    ax2.plot(epochs, clean_balanced_accs, label='Balanced')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Clean Validation Metrics')
    ax2.legend()
    ax2.grid(True)

    # Plot poisoned validation metrics
    ax3.plot(epochs, poisoned_true_positive_accs, label='True Positive')
    ax3.plot(epochs, poisoned_true_negative_accs, label='True Negative')
    ax3.plot(epochs, poisoned_balanced_accs, label='Balanced')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Poisoned Validation Metrics')
    ax3.legend()
    ax3.grid(True)

    # Plot attack success rates
    ax4.plot(epochs, attack_success_rates, label='Overall Attack Success Rate')
    ax4.plot(epochs, modified_label_attack_success_rates, label='Modified Label Success Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Backdoor Attack Success Rates')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(experiment_dir, 'metrics_sl1.png'))
    plt.close()


def main():
    # Setup experiment directory and logging
    experiment_dir = setup_experiment()

    # Set random seed for reproducibility
    set_seed()

    # === Load CelebA ===
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor()
    ])

    full_dataset = CelebA(root='../../../data', target_type='attr', download=False, transform=transform)

    # === Split into Train and Validation Sets ===
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=CONFIG['test_size'],
        stratify=full_dataset.attr[:, 20])

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Create poisoned semi-supervised dataset for training
    poisoned_train_data = PoisonedSemiSupervisedCelebA(train_dataset)

    # Create clean validation dataset
    clean_val_data = PoisonedSemiSupervisedCelebA(val_dataset, poison_ratio=0.0)

    # Create poisoned validation dataset with all samples poisoned
    poisoned_val_data = PoisonedSemiSupervisedCelebA(val_dataset, poison_ratio=1.0)

    # Create data loaders
    train_loader = DataLoader(poisoned_train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    clean_val_loader = DataLoader(clean_val_data, batch_size=CONFIG['batch_size'], shuffle=False)
    poisoned_val_loader = DataLoader(poisoned_val_data, batch_size=CONFIG['batch_size'], shuffle=False)

    model = CelebANet().to(CONFIG['device'])
    pos_weight = torch.ones([CONFIG['num_attributes']]) * CONFIG['bce_weight']
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    semantic_loss = SemanticLoss(CONFIG['sdd_path'], CONFIG['vtree_path']).to(CONFIG['device'])

    # === Training Loop ===
    for epoch in range(CONFIG['epochs']):
        train_model(model, train_loader, loss_fn, optimizer, semantic_loss, epoch)

        # Evaluate on clean and poisoned validation sets
        print("\nEvaluating on clean validation data:")
        evaluate_model(model, clean_val_loader, is_poisoned=False)
        print("\nEvaluating on poisoned validation data:")
        evaluate_model(model, poisoned_val_loader, is_poisoned=True)

    # Plot metrics
    plot_metrics(experiment_dir)

    # Show model predictions
    print("\nGenerating visualization examples...")
    clean_images, clean_labels, clean_preds = get_predictions(model, clean_val_loader, is_poisoned=False)
    plot_images(clean_images, clean_labels, clean_preds, "Semantic Loss Predictions on Clean Images",
                experiment_dir=experiment_dir)

    poisoned_images, poisoned_labels, poisoned_preds = get_predictions(model, poisoned_val_loader, is_poisoned=True)
    plot_images(poisoned_images, poisoned_labels, poisoned_preds, "Semantic Loss Predictions on Poisoned Images",
                experiment_dir=experiment_dir)


if __name__ == '__main__':
    main()
