import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import EmotionCNN


def evaluate_model(
    model_path: str = "models/best_model.pt",
    test_dir: str = "data/test",
    batch_size: int = 64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Évaluation sur : {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(test_dataset.classes)
    model = EmotionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    print("🔍 Test du modèle en cours...\n")
    progress_bar = tqdm(test_loader, desc="Évaluation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"\n✅ Précision finale : {accuracy:.2f}%")
    print(f"📉 Perte moyenne : {avg_loss:.4f}")

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10, 5))
    grid = utils.make_grid(images[:8])
    npimg = grid.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg.squeeze(), cmap="gray")
    plt.title("Prédictions : " + ", ".join([test_dataset.classes[p] for p in preds[:8]]))
    plt.axis("off")
    plt.show()

    print("📊 Évaluation terminée avec succès !")

    return {"test_loss": avg_loss, "test_acc": accuracy}


if __name__ == "__main__":
    evaluate_model()
