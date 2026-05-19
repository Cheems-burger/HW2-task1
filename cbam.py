import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import swanlab

swanlab.init(
    project="Flowers102-CBAM",
    config={
        "model": "ResNet18+CBAM",
        "dataset": "Flowers102",
        "batch_size": 64,
        "lr_backbone": 1e-4,
        "lr_cbam": 1e-3,
        "lr_classifier": 1e-3,
        "optimizer": "Adam",
        "epochs": 20,
        "unfreeze": "layer3+layer4"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        return x * spatial_att

class ResNet18WithCBAM(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.cbam = CBAM(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = ResNet18WithCBAM(num_classes=102).to(device)

for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[6].parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.features[6].parameters(), "lr": 1e-4},
    {"params": model.features[7].parameters(), "lr": 1e-4},
    {"params": model.cbam.parameters(), "lr": 1e-3},
    {"params": model.classifier.parameters(), "lr": 1e-3}
])

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct, train_total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:2d}/{num_epochs} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    swanlab.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }, step=epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_cbam.pth")

print(f"Best validation accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load("best_model_cbam.pth"))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")

swanlab.finish()