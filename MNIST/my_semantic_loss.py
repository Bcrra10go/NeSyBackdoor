import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss

# === Load constraints ===
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

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

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()
semantic_loss = SemanticLoss('constraint.sdd', 'constraint.vtree')

# === Training ===
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        labels_bin = torch.zeros(images.size(0), 10)
        labels_bin.scatter_(1, labels.unsqueeze(1), 1.0)

        preds = model(images)

        loss_bce = loss(preds, labels_bin)

        # Add semantic loss to enforce one-hot
        preds_reshaped = preds.view(-1, 1, 10) # Reshape to 10 variables with 2 states
        loss_sem = semantic_loss(preds_reshaped)

        # Combine both losses
        loss_sum = loss_bce + 0.1 * loss_sem
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        total_loss += loss_sum.item()

    print(f"Epoch {epoch + 1}, BCE: {loss_bce.item():.4f}, Semantic: {loss_sem.item():.4f}, Total: {loss_sum.item():.4f}")

    # === Evaluate on Test Set ===
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")