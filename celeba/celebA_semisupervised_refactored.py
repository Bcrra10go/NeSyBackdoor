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
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset

# Hyperparameters and Configuration
CONFIG = {
    'random_seed': 42,
    'batch_size': 64,
    'epochs': 5,
    'test_size' : 0.2,
    'num_attributes' : 40,
    'labeled_ratio' : 0.4,
    'learning_rate': 0.001,
    'bce_weight': 3,
    'sl_weight': 0.3,
    'threshold' : 0.3,
    'sdd_path' : 'constraints/celebA.sdd',
    'vtree_path' : 'constraints/celebA.vtree',
    'num_examples': 5,  # Number of examples to show in visualizations
    'attr_names' : ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                  'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                  'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                  'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                  'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                  'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
}

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

# === Dataset ===
class SemiSupervisedCelebA(Dataset):
    def __init__(self, dataset, labeled_ratio=CONFIG['labeled_ratio']):
        self.dataset = dataset

        # Get subset indices
        subset_indices = dataset.indices
        full_attrs = dataset.dataset.attr.clone()  # shape: [N, 40]

        # Only use attributes for the subset
        self.attrs = full_attrs[subset_indices]
        self.attrs[self.attrs == -1] = 0

        # Create label mask
        self.labeled_mask = torch.zeros(len(self.attrs), dtype=torch.bool)
        labeled_count = int(len(self.attrs) * labeled_ratio)
        labeled_indices = torch.randperm(len(self.attrs))[:labeled_count]
        self.labeled_mask[labeled_indices] = True

        # Mask unlabeled attributes by setting to -1
        self.attrs[~self.labeled_mask] = -1
        
        # Move tensors to device
        self.attrs = self.attrs.to(CONFIG['device'])
        self.labeled_mask = self.labeled_mask.to(CONFIG['device'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.attrs[idx]  # shape: [40], with values 0, 1, or -1
        has_label = self.labeled_mask[idx]
        return image, label, has_label

# === Model ===
class CelebANet(nn.Module):
    def __init__(self, num_attrs=40):
        super(CelebANet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3, 512),
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
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
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

def train_model(model, train_loader, loss_fn, optimizer, semantic_loss, epoch, epochs=CONFIG['epochs']):
    model.train()
    # total_loss = 0
    # total_bce_loss = 0
    # total_sem_loss = 0
    # labeled = 0
    # unlabeled = 0

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
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        # Update statistics
        # total_loss += loss_sum.item()
        # total_bce_loss += loss_bce.item()
        # total_sem_loss += loss_sem.item()

    #     num_labeled = is_labeled.sum().item()
    #     num_unlabeled = (~is_labeled).sum().item()
    #
    #     labeled += num_labeled
    #     unlabeled += num_unlabeled
    #
    # # Print detailed statistics
    # avg_loss = total_loss / len(train_loader)
    # avg_bce = total_bce_loss / len(train_loader)
    # avg_sem = total_sem_loss / len(train_loader)
    #
    # print(f"Epoch {epoch + 1}/{epochs}")
    # print(f"  Training Loss: {avg_loss:.4f}")
    # print(f"  BCE Loss: {avg_bce:.4f}")
    # print(f"  Semantic Loss: {avg_sem:.4f}")
    # print(f"  Labeled: {labeled}, Unlabeled: {unlabeled}")


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    # Initialize counters for each attribute
    true_positives = torch.zeros(CONFIG['num_attributes'])  # Correctly predicted 1s
    true_negatives = torch.zeros(CONFIG['num_attributes'])  # Correctly predicted 0s
    total_positives = torch.zeros(CONFIG['num_attributes'])  # Total actual 1s
    total_negatives = torch.zeros(CONFIG['num_attributes'])  # Total actual 0s
    total_samples = torch.zeros(CONFIG['num_attributes'])  # Total samples per attribute
    
    threshold = CONFIG['threshold']
    
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(loader):
            images = images.to(CONFIG['device'])
            attrs = attrs.to(CONFIG['device'])
            attrs[attrs == -1] = 0
            
            preds = torch.sigmoid(model(images))  # Apply sigmoid to raw outputs
            preds_binary = (preds > threshold).float()
    #         correct += (preds_binary == attrs).sum().item()
    #         total += torch.numel(attrs)
    #
    # accuracy = 100 * correct / total
    # print(f"Validation Accuracy: {accuracy:.4f}")
            
            # Update per-attribute statistics
            for i in range(CONFIG['num_attributes']):
                # True positives and negatives
                true_positives[i] += ((preds_binary[:, i] == 1) & (attrs[:, i] == 1)).sum().item()
                true_negatives[i] += ((preds_binary[:, i] == 0) & (attrs[:, i] == 0)).sum().item()
                # Total actual positives and negatives
                total_positives[i] += (attrs[:, i] == 1).sum().item()
                total_negatives[i] += (attrs[:, i] == 0).sum().item()
                # Total samples
                total_samples[i] += attrs[:, i].size(0)

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
        
        # Calculate true positive rate (sensitivity) and true negative rate (specificity)
        tpr = (tp / total_pos) if total_pos > 0 else 0  # True Positive Rate (Sensitivity)
        tnr = (tn / total_neg) if total_neg > 0 else 0  # True Negative Rate (Specificity)
        
        # Balanced accuracy is the average of TPR and TNR
        balanced_acc = 0.5 * (tpr + tnr)
        
        attr_name = CONFIG['attr_names'][i]
        print(f"{attr_name:<20} {balanced_acc:>7.2%}        {tpr:>7.2%}         {tnr:>7.2%}     {int(tp):>4d}   {int(tn):>4d}   {int(total_pos):>4d}    {int(total_neg):>4d}")
        
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
    print(f"Average Balanced Accuracy: {avg_balanced_acc:.2%}")
    
    if total_positives.sum() == 0:
        print("\nWARNING: No positive samples in the evaluation set!")

def main():
    # Set random seed for reproducibility
    set_seed()

    # === Load CelebA ===
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to manageable size
        transforms.ToTensor()
    ])

    full_dataset = CelebA(root='data', target_type='attr', download=False, transform=transform)

    # === Split into Train and Validation Sets ===
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=CONFIG['test_size'],
        stratify=full_dataset.attr[:, 20])

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    semi_supervised_dataset = SemiSupervisedCelebA(train_dataset, CONFIG['labeled_ratio'])

    train_loader = DataLoader(semi_supervised_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = CelebANet().to(CONFIG['device'])  # Move model to GPU
    pos_weight = torch.ones([CONFIG['num_attributes']]) * CONFIG['bce_weight']  # weight > 1 to focus on positive samples
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(CONFIG['device'])  # Use weighted loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    semantic_loss = SemanticLoss(CONFIG['sdd_path'], CONFIG['vtree_path']).to(CONFIG['device'])  # Move semantic loss to GPU

    # === Training Loop ===
    for epoch in range(CONFIG['epochs']):
        train_model(model, train_loader, loss_fn, optimizer, semantic_loss, epoch)
        evaluate_model(model, val_loader)

    # Show model predictions on clean and poisoned images
    images, labels, preds = get_predictions(model, val_loader)
    plot_images(images, labels, preds, "Model Predictions on Clean Images")

if __name__ == '__main__':
    main()