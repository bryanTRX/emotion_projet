import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import EmotionCNN

pt_path = "best_model.pt"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root="../data/train_augmented", transform=transform)
test_dataset = datasets.ImageFolder(root="../data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = EmotionCNN()

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"✅ Utilisation du device : {device}")
model.to(device)
print(f"Modèle envoyé sur : {device}")



if os.path.exists(pt_path):
    model.load_state_dict(torch.load(pt_path, map_location=device))
    print(f"✅ Modèle existant chargé depuis {pt_path}, entraînement continu")
else:
    print("✅ Nouveau modèle, entraînement depuis zéro")



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
    
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
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

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}% 🎯")       

torch.save(model.state_dict(), pt_path)
print("✅ Modèle sauvegardé !")

model.eval()

onnx_path="best_model.onnx"
dummy_input = torch.randn(1, 1, 48, 48, device=device)
torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
print(f"✅ Modèle exporté en ONNX : {onnx_path}")




