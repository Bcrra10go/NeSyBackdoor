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

        # Access the original constraints dataset
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

# Load full constraints
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Split into train and validation sets
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=full_dataset.targets,
)

# Validation set: remain fully labeled
val_dataset = Subset(full_dataset, val_indices)

train_set = SemiSupervisedSplitMNIST(Subset(full_dataset, train_indices), labeled_ratio=0.05)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()
# semantic_loss = SemanticLoss('constraints/no_constraint_MNIST.sdd', 'constraints/no_constraint_MNIST.vtree')
semantic_loss = SemanticLoss('constraints/one_hot_MNIST.sdd', 'constraints/constraints/one_hot_MNIST.vtree')

for epoch in range(5):
    model.train()
    total_loss = 0
    labeled = 0
    unlabeled = 0

    for images, labels in train_loader:
        preds = model(images)  # (batch, 10), outputs in [0, 1] due to sigmoid

        # === Separate labeled and unlabeled samples ===
        is_labeled = labels != -1
        loss_bce = torch.tensor(0.0)  # default

        if is_labeled.any():
            labeled_preds = preds[is_labeled]
            labeled_labels = labels[is_labeled]
            labels_bin = torch.nn.functional.one_hot(labeled_labels, num_classes=10).float()

            # BCE on labeled samples
            loss_bce = loss(labeled_preds, labels_bin)

        # === Semantic loss on all samples ===
        preds_reshaped = preds.view(-1, 1, 10)  # Reshape to 10 variables
        loss_sem = semantic_loss(preds_reshaped)

        # === Combine losses ===
        loss_sum = loss_bce + 0.3 * loss_sem
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        num_labeled = is_labeled.sum().item()
        num_unlabeled = (~is_labeled).sum().item()

        labeled += num_labeled
        unlabeled += num_unlabeled

    print(f"Epoch {epoch + 1}, BCE: {loss_bce.item():.4f}, Semantic: {loss_sem.item():.4f}, Total: {loss_sum.item():.4f}, Labeled: {labeled}, Unlabeled: {unlabeled}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images, labels
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
