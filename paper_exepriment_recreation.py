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
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
            nn.Sigmoid()  # For semantic loss
        )

    def forward(self, x):
        return self.net(x)

# HYPERPARAMETERS
sl_weight = 0.3
test_size = 0.2
labeled_ratio = 0.001666
lr = 0.001
epochs = 20
batch_size = 10

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

# === Model: With Semantic Loss ===
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()
semantic_loss = SemanticLoss('constraints/one_hot_MNIST.sdd', 'constraints/one_hot_MNIST.vtree')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        ### === Model with One Hot Constraints ===
        preds = model(images)  # (batch, 10), outputs in [0, 1] due to sigmoid

        # === Separate labeled and unlabeled samples ===
        is_labeled = labels != -1
        loss_bce = torch.tensor(0.0)  # default

        if is_labeled.any():
            # One-hot encode only the labeled labels
            labels_bin = torch.zeros_like(preds)
            labels_bin[is_labeled] = torch.nn.functional.one_hot(labels[is_labeled], num_classes=10).float()

            # BCE on labeled samples
            loss_bce = loss_fn(preds, labels_bin)

        # === Semantic loss on all samples ===
        preds_reshaped = preds.view(-1, 1, 10)  # Reshape to 10 variables
        loss_sem = semantic_loss(preds_reshaped)

        # === Combine losses ===
        loss_sum = loss_bce + sl_weight * loss_sem
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        total_loss += loss_sum


    print(f"Epoch {epoch+1} | With One Hot Encoding: {total_loss:.4f}")

    # === Evaluate Model WITH One Hot ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
