"""
Real-world example: Comparing two similar CNN models on MNIST.
This demonstrates how a deeper model can outperform a shallow one.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pravi_torch_view.model_compare import compare_models


# Simple CNN with 1 conv layer
class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x


# Deeper CNN with 2 conv layers
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(model, train_loader, test_loader, epochs=5):
    """Train a model and return its final accuracy and loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def main():
    print("Generating synthetic MNIST-like data...")

    # Create synthetic data (28x28 images, 10 classes)
    torch.manual_seed(42)
    train_data = torch.randn(1000, 1, 28, 28)
    train_labels = torch.randint(0, 10, (1000,))
    test_data = torch.randn(200, 1, 28, 28)
    test_labels = torch.randint(0, 10, (200,))

    train_loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_data, test_labels),
        batch_size=32
    )

    # Initialize models
    print("\nTraining Shallow CNN (1 conv layer)...")
    shallow_model = ShallowCNN()
    shallow_acc, shallow_loss = train_and_evaluate(
        shallow_model, train_loader, test_loader, epochs=10
    )

    print("Training Deep CNN (2 conv layers)...")
    deep_model = DeepCNN()
    deep_acc, deep_loss = train_and_evaluate(
        deep_model, train_loader, test_loader, epochs=10
    )

    # Prepare results for comparison
    results = {
        'Shallow CNN': {
            'accuracy': shallow_acc,
            'loss': shallow_loss,
            'params': count_parameters(shallow_model),
            'layers': 1,
        },
        'Deep CNN': {
            'accuracy': deep_acc,
            'loss': deep_loss,
            'params': count_parameters(deep_model),
            'layers': 2,
        }
    }

    # Visualize comparison
    compare_models(results)


if __name__ == '__main__':
    main()
