import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EmotionCNN


def get_dataloader(train_dir: str, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def get_device(force_cuda: bool = True) -> torch.device:
    print("🔍 Vérification CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif force_cuda:
        raise RuntimeError("❌ CUDA non disponible. Vérifie ton installation de PyTorch GPU.")
    elif torch.backends.mps.is_available():
        print("⚙️ Utilisation de Metal Performance Shaders (MPS) sur macOS.")
        return torch.device("mps")
    else:
        print("🧩 Aucune accélération matérielle détectée. Utilisation du CPU.")
        return torch.device("cpu")


def init_model(num_classes: int = 7, pretrained_path: str | None = None) -> tuple[nn.Module, torch.device]:
    device = get_device(force_cuda=True)
    model = EmotionCNN(num_classes).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"✅ Modèle pré-entraîné chargé depuis : {pretrained_path}")

    return model, device


def train_model(model: nn.Module, device: torch.device, loader: DataLoader, epochs: int = 30) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}", acc=f"{100*correct/total:.2f}%")

        epoch_loss = total_loss / len(loader)
        epoch_acc = 100 * correct / total
        scheduler.step(epoch_loss)

        print(f"🎯 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    save_path = "models/best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modèle sauvegardé : {save_path}")


if __name__ == "__main__":
    loader = get_dataloader("data/train_augmented")
    model, device = init_model(num_classes=7)
    train_model(model, device, loader)
