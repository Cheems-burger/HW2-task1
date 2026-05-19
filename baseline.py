import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import swanlab

swanlab.init(
    project="Flowers102-Baseline",
    experiment_name="Baseline_pretrain_finetune",
    config={
        "model": "ResNet18",
        "pretrained": True,
        "unfreeze": "layer4",
        "lr_backbone": 1e-4,
        "lr_fc": 1e-3,
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

# 原始预训练模型预测（可选，保留原逻辑）
weights = ResNet18_Weights.IMAGENET1K_V1
imagenet_labels = weights.meta["categories"]
model_original = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
img, label = train_dataset[0]
model_original = model_original.to(device)
model_original.eval()
img_gpu = img.unsqueeze(0).to(device)
with torch.no_grad():
    output = model_original(img_gpu)
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_idx = probs.topk(5)
print("ResNet18 (ImageNet) predictions for a flower:")
for i in range(5):
    idx = top5_idx[0][i].item()
    prob = top5_probs[0][i].item()
    print(f"  {imagenet_labels[idx]:>30s}: {prob:.1%}")

model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model_ft.parameters():
    param.requires_grad = False
for param in model_ft.layer4.parameters():
    param.requires_grad = True

model_ft.fc = nn.Linear(model_ft.fc.in_features, 102)
model_ft = model_ft.to(device)

optimizer_ft = optim.Adam([
    {"params": model_ft.layer4.parameters(), "lr": 1e-4},
    {"params": model_ft.fc.parameters(), "lr": 1e-3}
])

criterion = nn.CrossEntropyLoss()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model_ft.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_ft.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    train_acc = train_correct / train_total

    model_ft.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
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
        torch.save(model_ft.state_dict(), "best_model_baseline.pth")

print(f"Best validation accuracy: {best_val_acc:.4f}")

model_ft.load_state_dict(torch.load("best_model_baseline.pth"))
model_ft.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")

swanlab.log({"best_val_acc": best_val_acc, "test_acc": test_acc})
swanlab.finish()

