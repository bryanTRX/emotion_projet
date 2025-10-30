import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import EmotionCNN

def get_dataloader(train_dir: str, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return train_loader

def init_model(num_classes: int = 7, pretrained_path: str | None = None) -> tuple[EmotionCNN, torch.device]:
    model = EmotionCNN(num_classes=num_classes)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"✅ Modèle chargé depuis {pretrained_path}")
    return model, device

def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 15
) -> None:
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=f"{running_loss/(loop.n+1):.4f}", acc=f"{100*correct/total:.2f}%")
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100*correct/total:.2f}%")

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"✅ Modèle sauvegardé : {path}")

if __name__ == "__main__":
    train_loader: DataLoader = get_dataloader("data/train_augmented")
    model, device = init_model(num_classes=7, pretrained_path="models/best_model.pth")
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    train_model(model, device, train_loader, criterion, optimizer, epochs=10)
    save_model(model, "models/best_model.pth")
