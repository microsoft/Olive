# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, resnet50

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 50
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose(
    [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
)

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


def load_resnet_model():
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Sequential(torch.nn.Linear(2048, 64), torch.nn.ReLU(inplace=True), torch.nn.Linear(64, 10))
    return resnet


model = load_resnet_model().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item())
            )

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy of the model on the test images: {} %".format(100 * correct / total))

# Save the model
model.to("cpu")
torch.save(model, "models/resnet_trained_for_cifar10.pt")
