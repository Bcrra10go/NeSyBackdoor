import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss
from sklearn.model_selection import train_test_split

class SemiSupervisedSplitMNIST(Dataset):
    def __init__(self, dataset, labeled_ratio=0.1):
        self.dataset = dataset

        # Access the original MNIST dataset
        full_data = dataset.dataset.data
        full_targets = dataset.dataset.targets.clone()

        # Get the subset indices
        subset_indices = dataset.indices

        # Use only data/targets from the subset
        self.data = full_data[subset_indices]
        self.targets = full_targets[subset_indices]

        # Mask to keep track of labeled samples
        self.labeled_mask = torch.zeros(len(self.targets), dtype=torch.bool)
        labeled_count = int(len(self.targets) * labeled_ratio)
        labeled_indices = torch.randperm(len(self.targets))[:labeled_count]
        self.labeled_mask[labeled_indices] = True

        # Set unlabeled targets to -1
        self.targets[~self.labeled_mask] = -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.targets[idx]
        return image, label

# === Model ===
class MNISTNet(nn.Module): # Model (MNIST 784 -> 128 (ReLu) -> 10 (Sigmoid))
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

# HYPERPARAMETERS
sl_weight = 0.3
test_size = 0.2
labeled_ratio = 0.002
lr = 0.001
epochs = 5
batch_size = 64

# Load full MNIST
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Split into train and validation sets
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=test_size,
    stratify=full_dataset.targets,
)

# Validation set: remain fully labeled
val_dataset = Subset(full_dataset, val_indices)

train_set = SemiSupervisedSplitMNIST(Subset(full_dataset, train_indices), labeled_ratio=labeled_ratio)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model 1: With One Hot Encoding ===
model_one_hot = MNISTNet()
optimizer_one_hot = optim.Adam(model_one_hot.parameters(), lr=lr)
loss_fn_one_hot = nn.BCELoss()
semantic_loss_one_hot = SemanticLoss('constraints/one_hot_MNIST.sdd', 'constraints/one_hot_MNIST.vtree')

# === Model 2: Without One Hot Encoding ===
model_no_constraint = MNISTNet()
optimizer_no_constraint = optim.Adam(model_no_constraint.parameters(), lr=lr)
loss_fn_no_constraint = nn.BCELoss()
semantic_loss_no_constraint = SemanticLoss('constraints/no_constraint_MNIST.sdd', 'constraints/no_constraint_MNIST.vtree')

for epoch in range(epochs):
    model_one_hot.train()
    model_no_constraint.train()
    total_loss_one_hot = 0
    total_loss_no_constraint = 0

    for images, labels in train_loader:
        ### === Model with One Hot Constraints ===
        preds_one_hot = model_one_hot(images)  # (batch, 10), outputs in [0, 1] due to sigmoid

        # === Separate labeled and unlabeled samples ===
        is_labeled_one_hot = labels != -1
        loss_bce_one_hot = torch.tensor(0.0)  # default

        if is_labeled_one_hot.any():
            # One-hot encode only the labeled labels
            labels_bin = torch.zeros_like(preds_one_hot)
            labels_bin[is_labeled_one_hot] = torch.nn.functional.one_hot(labels[is_labeled_one_hot], num_classes=10).float()

            # BCE on labeled samples
            loss_bce_one_hot = loss_fn_one_hot(preds_one_hot, labels_bin)

        # === Semantic loss on all samples ===
        preds_one_hot_reshaped = preds_one_hot.view(-1, 1, 10)  # Reshape to 10 variables
        loss_sem_one_hot = semantic_loss_one_hot(preds_one_hot_reshaped)

        # === Combine losses ===
        loss_sum_one_hot = loss_bce_one_hot + sl_weight * loss_sem_one_hot
        optimizer_one_hot.zero_grad()
        loss_sum_one_hot.backward()
        optimizer_one_hot.step()

        total_loss_one_hot += loss_sum_one_hot

        ### === Model without One Hot Constraints ===
        preds_no_constraint = model_no_constraint(images)  # (batch, 10), outputs in [0, 1] due to sigmoid

        # === Separate labeled and unlabeled samples ===
        is_labeled_no_contraint = labels != -1
        loss_bce_no_constraint = torch.tensor(0.0)  # default

        if is_labeled_no_contraint.any():
            # One-hot encode only the labeled labels
            labels_bin = torch.zeros_like(preds_no_constraint)
            labels_bin[is_labeled_no_contraint] = torch.nn.functional.one_hot(labels[is_labeled_no_contraint], num_classes=10).float()

            # BCE on labeled samples
            loss_bce_no_constraint = loss_fn_no_constraint(preds_no_constraint, labels_bin)

        # === Semantic loss on all samples ===
        preds_reshaped_no_constraint = preds_no_constraint.view(-1, 1, 10)  # Reshape to 10 variables
        loss_sem_no_constraint = semantic_loss_no_constraint(preds_reshaped_no_constraint)

        # === Combine losses ===
        loss_sum_no_constraint = loss_bce_no_constraint + sl_weight * loss_sem_no_constraint
        optimizer_no_constraint.zero_grad()
        loss_sum_no_constraint.backward()
        optimizer_no_constraint.step()

        total_loss_no_constraint += loss_sum_no_constraint


    print(f"Epoch {epoch+1} | With One Hot Encoding: {total_loss_one_hot:.4f} | Without One Hot Encoding: {total_loss_no_constraint:.4f}")

    # === Evaluate Model WITH One Hot ===
    model_one_hot.eval()
    correct_one_hot = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model_one_hot(images)
            predicted = torch.argmax(outputs, dim=1)
            correct_one_hot += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy_one_hot = 100 * correct_one_hot / total
    print(f"Test Accuracy (with one hot encoding): {accuracy_one_hot:.2f}%")

    # === Evaluate Model WITHOUT One Hot ===
    model_no_constraint.eval()
    correct_no_constraint = 0
    total = 0
    with (torch.no_grad()):
        for images, labels in val_loader:
            outputs = model_no_constraint(images)
            predicted = torch.argmax(outputs, dim=1)
            correct_no_constraint += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy_no_constraint = 100 * correct_no_constraint / total
    print(f"Test Accuracy (without one hot encoding): {accuracy_no_constraint:.2f}%")
