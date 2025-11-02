import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import EmotionCNN

EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "suprised"
]

def get_device(force_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        print(f"✅ CUDA activé : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif force_cuda:
        raise RuntimeError("❌ CUDA non disponible sur cette machine.")
    elif torch.backends.mps.is_available():
        print("⚙️ Utilisation du backend MPS (Mac).")
        return torch.device("mps")
    else:
        print("🧩 Utilisation du CPU.")
        return torch.device("cpu")

def load_model(model_path: str, num_classes: int = 7) -> tuple[nn.Module, torch.device]:
    device = get_device(force_cuda=False)
    
    if model_path.endswith('_traced.pt'):
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = EmotionCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    print(f"✅ Modèle chargé depuis : {model_path}")
    return model, device

def preprocess_image(img) -> torch.Tensor:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(img).unsqueeze(0)

def get_emotion_label(pred_idx: int) -> str:
    if 0 <= pred_idx < len(EMOTION_LABELS):
        return EMOTION_LABELS[pred_idx]
    return "Inconnu"
