import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import EmotionCNN

def evaluate_model(model_path: str = "models/best_model.pth", test_dir: str = "data/test", batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device utilisé : {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total, correct, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Évaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Accuracy : {acc:.2f}% | Loss moyenne : {loss_sum/len(loader):.4f}")

    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    grid = utils.make_grid(images[:8])
    npimg = grid.numpy().transpose(1, 2, 0)
    plt.imshow(npimg.squeeze(), cmap="gray")
    plt.title("Prédictions : " + ", ".join([dataset.classes[p] for p in preds[:8]]))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
