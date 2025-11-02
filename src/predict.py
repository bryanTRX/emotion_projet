import torch
from PIL import Image
from utils import load_model, preprocess_image, get_emotion_label

def predict_image(image_path: str, model_path: str = "models/best_model_traced.pt") -> str:
    model, device = load_model(model_path)
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        label = get_emotion_label(pred.item())

    print(f"🧠 Émotion prédite : {label}")
    return label

if __name__ == "__main__":
    predict_image("")
