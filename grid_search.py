import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import swanlab
import time

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

def train_and_evaluate(config, num_epochs=20):
    print(f"\n实验: {config['name']}")
    print(f"  backbone lr: {config['lr_backbone']}, fc lr: {config['lr_fc']}")
    print(f"  优化器: {config['optimizer']}, 解冻层: {config['unfreeze']}")

    swanlab.init(
        project="Flowers102-GridSearch",
        experiment_name=config['name'],
        config={
            "lr_backbone": config['lr_backbone'],
            "lr_fc": config['lr_fc'],
            "optimizer": config['optimizer'],
            "unfreeze": config['unfreeze'],
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "model": "ResNet18"
        }
    )

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    if config['unfreeze'] == 'layer4':
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif config['unfreeze'] == 'layer3+layer4':
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif config['unfreeze'] == 'none':
        pass 

    model.fc = nn.Linear(model.fc.in_features, 102)
    model = model.to(device)

    if config['unfreeze'] != 'none':
        if config['unfreeze'] == 'layer4':
            backbone_params = model.layer4.parameters()
        else:
            backbone_params = list(model.layer3.parameters()) + list(model.layer4.parameters())

        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam([
                {"params": backbone_params, "lr": config['lr_backbone']},
                {"params": model.fc.parameters(), "lr": config['lr_fc']}
            ])
        else:  
            optimizer = optim.SGD([
                {"params": backbone_params, "lr": config['lr_backbone'], "momentum": 0.9},
                {"params": model.fc.parameters(), "lr": config['lr_fc'], "momentum": 0.9}
            ])
    else:
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.fc.parameters(), lr=config['lr_fc'])
        else:
            optimizer = optim.SGD(model.fc.parameters(), lr=config['lr_fc'], momentum=0.9)

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
    print(f"Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

    swanlab.log({"best_val_acc": best_val_acc, "test_acc": test_acc})
    swanlab.finish()

    return {
        'name': config['name'],
        'lr_backbone': config['lr_backbone'],
        'lr_fc': config['lr_fc'],
        'optimizer': config['optimizer'],
        'unfreeze': config['unfreeze'],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }

configs = [
    {"name": "B_small_backbone_lr", "lr_backbone": 1e-5, "lr_fc": 1e-3, "optimizer": "Adam", "unfreeze": "layer4"},
    {"name": "C_unified_lr", "lr_backbone": 1e-4, "lr_fc": 1e-4, "optimizer": "Adam", "unfreeze": "layer4"},
    {"name": "D_SGD", "lr_backbone": 1e-4, "lr_fc": 1e-3, "optimizer": "SGD", "unfreeze": "layer4"},
    {"name": "E_freeze_all", "lr_backbone": 0, "lr_fc": 1e-3, "optimizer": "Adam", "unfreeze": "none"},
    {"name": "F_unfreeze_layer3_4", "lr_backbone": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "unfreeze": "layer3+layer4"},
]

results = []
for cfg in configs:
    result = train_and_evaluate(cfg, num_epochs=20)
    results.append(result)
    time.sleep(5)

for r in results:
    print(f"{r['name']}: Val Acc = {r['best_val_acc']:.4f}, Test Acc = {r['test_acc']:.4f}")