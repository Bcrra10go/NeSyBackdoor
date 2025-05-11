import torch
import torch.nn as nn
import torchvision.transforms as transforms
from semantic_loss_pytorch import SemanticLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import optim
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset


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
SL_WEIGHT = 0.1

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = CelebANet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
semantic_loss = SemanticLoss('constraints/celebA.sdd', 'constraints/celebA.vtree')

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, attrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
        preds = model(images)
        loss = loss_fn(preds, attrs.float())

        preds_reshaped = preds.view(-1, 1, 40)  # Reshape to 40 variables
        loss_sem = semantic_loss(preds_reshaped)

        # print(f"BCE Loss:  {loss:.4f}, Semantic Loss: {loss_sem:.4f}, Weighted: {loss + (loss_sem * SL_WEIGHT):.4f}")

        loss_sum = loss + SL_WEIGHT * loss_sem
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        total_loss += loss_sum.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss / len(train_loader):.4f}")

    # === Evaluation ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, attrs in val_loader:
            preds = model(images)
            preds_binary = (preds > 0.5).float()
            correct += (preds_binary == attrs).sum().item()
            total += torch.numel(attrs)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
