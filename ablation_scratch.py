import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import swanlab

# 初始化 SwanLab
swanlab.init(
    project="Flowers102-Ablation",
    experiment_name="Random_Init_Scratch",
    config={
        "model": "ResNet18",
        "pretrained": False,
        "freeze": "none",
        "lr": 1e-3,
        "optimizer": "Adam",
        "batch_size": 64,
        "epochs": 20,
        "dataset": "Flowers102"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.Flowers102(
    root="./data", split="train", download=False, transform=train_transform
)
val_dataset = torchvision.datasets.Flowers102(
    root="./data", split="val", download=False, transform=test_transform
)
test_dataset = torchvision.datasets.Flowers102(
    root="./data", split="test", download=False, transform=test_transform
)

print("Training:", len(train_dataset), "images")
print("Validation:", len(val_dataset), "images")
print("Test:", len(test_dataset), "images")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:2d}/{num_epochs} Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")

    swanlab.log({
        "train_loss": train_loss/train_total,
        "train_acc": train_acc,
        "val_loss": val_loss/val_total,
        "val_acc": val_acc
    }, step=epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_scratch.pth")

print(f"Best validation accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load("best_model_scratch.pth"))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")

swanlab.log({"best_val_acc": best_val_acc, "test_acc": test_acc})
swanlab.finish()