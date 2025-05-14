import torch
import torch.nn as nn
import torchvision.transforms as transforms
from semantic_loss_pytorch import SemanticLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import optim
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset

# === Dataset ===
class SemiSupervisedCelebA(Dataset):
    def __init__(self, dataset, labeled_ratio=0.1):
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
            nn.Linear(512, num_attrs),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Hyperparameters ===
BATCH_SIZE = 64
EPOCHS = 5
NUM_ATTRS = 40
TEST_SIZE = 0.2
SL_WEIGHT = 0.3
LABELED_RATIO = 0.02
LR = 0.001

# === Load CelebA ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to manageable size
    transforms.ToTensor()
])

full_dataset = CelebA(root='./data', target_type='attr', download=False, transform=transform)

# === Split into Train and Validation Sets ===
train_idx, val_idx = train_test_split(
    range(len(full_dataset)),
    test_size=TEST_SIZE,
    stratify=full_dataset.attr[:, 20])

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
semi_supervised_dataset = SemiSupervisedCelebA(train_dataset, LABELED_RATIO)

train_loader = DataLoader(semi_supervised_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = CelebANet()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()
semantic_loss = SemanticLoss('constraints/celebA.sdd', 'constraints/celebA.vtree')

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    labeled = 0
    unlabeled = 0
    for images, attrs, is_labeled in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
        preds = model(images)

        loss_bce = torch.tensor(0.0)  # default

        if is_labeled.any():
            labeled_preds = preds[is_labeled]
            labeled_attrs = attrs[is_labeled].float()
            loss_bce = loss_fn(labeled_preds, labeled_attrs)

        preds_reshaped = preds.view(-1, 1, 40)  # Reshape to 40 variables
        loss_sem = semantic_loss(preds_reshaped)

        loss_sum = loss_bce + SL_WEIGHT * loss_sem
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        total_loss += loss_sum.item()

        num_labeled = is_labeled.sum().item()
        num_unlabeled = (~is_labeled).sum().item()

        labeled += num_labeled
        unlabeled += num_unlabeled

    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss / len(train_loader):.4f}, Labeled: {labeled}, Unlabeled: {unlabeled}")

    # === Evaluation ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, attrs in val_loader:
            attrs[attrs == -1] = 0
            preds = model(images)
            preds_binary = (preds > 0.5).float()
            correct += (preds_binary == attrs).sum().item()
            total += torch.numel(attrs)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")