import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss

# === Load MNIST ===
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# === Model ===
class MNISTNet(nn.Module):
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

# === Model 1: With Semantic Loss ===
model_sem = MNISTNet()
optimizer_sem = optim.Adam(model_sem.parameters(), lr=0.001)
loss_fn_sem = nn.BCELoss()
semantic_loss = SemanticLoss('constraint.sdd', 'constraint.vtree')

# === Model 2: Without Semantic Loss ===
model_plain = MNISTNet()
optimizer_plain = optim.Adam(model_plain.parameters(), lr=0.001)
loss_fn_plain = nn.BCELoss()

for epoch in range(5):
    total_loss_sem = 0
    total_loss_plain = 0
    for images, labels in train_loader:
        labels_bin = torch.zeros(images.size(0), 10)
        labels_bin.scatter_(1, labels.unsqueeze(1), 1.0)

        ### === Model with Semantic Loss ===
        preds_sem = model_sem(images)
        loss_bce_sem = loss_fn_sem(preds_sem, labels_bin)
        loss_sem_term = semantic_loss(preds_sem.view(-1, 1, 10))
        loss_total_sem = loss_bce_sem + 0.1 * loss_sem_term

        optimizer_sem.zero_grad()
        loss_total_sem.backward()
        optimizer_sem.step()

        total_loss_sem += loss_total_sem.item()

        ### === Model without Semantic Loss ===
        preds_plain = model_plain(images)
        loss_plain = loss_fn_plain(preds_plain, labels_bin)

        optimizer_plain.zero_grad()
        loss_plain.backward()
        optimizer_plain.step()

        total_loss_plain += loss_plain.item()

    print(f"Epoch {epoch+1} | With Semantic: {total_loss_sem:.4f} | Without Semantic: {total_loss_plain:.4f}")

    # === Evaluate on Test Set ===
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # === Evaluate Model WITH Semantic Loss ===
    model_sem.eval()
    correct_sem = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_sem(images)
            predicted = torch.argmax(outputs, dim=1)
            correct_sem += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy_sem = 100 * correct_sem / total
    print(f"Test Accuracy (with semantic loss): {accuracy_sem:.2f}%")

    # === Evaluate Model WITHOUT Semantic Loss ===
    model_plain.eval()
    correct_plain = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_plain(images)
            predicted = torch.argmax(outputs, dim=1)
            correct_plain += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy_plain = 100 * correct_plain / total
    print(f"Test Accuracy (without semantic loss): {accuracy_plain:.2f}%")
