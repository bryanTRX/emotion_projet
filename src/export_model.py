import torch
from src.model import EmotionCNN

model = EmotionCNN()
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()

example_input = torch.randn(1, 1, 48, 48)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("models/best_model_traced.pt")
print("✅ Modèle exporté : models/best_model_traced.pt")