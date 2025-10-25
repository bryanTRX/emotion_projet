import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import EmotionCNN


import torch
import torch.nn as nn

dummy_input = torch.randn(1, 1, 48, 48)
print(f"Input shape: {dummy_input.shape}")

model = EmotionCNN()
model.eval()

x = dummy_input
print("=== Conv layers ===")
for i, layer in enumerate(model.conv_layers):
    x = layer(x)
    print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")

print("\n=== Fully connected layers ===")
for i, layer in enumerate(model.fc_layers):
    x = layer(x)
    print(f"After FC layer {i} ({layer.__class__.__name__}): {x.shape}")
    
    
output = model(dummy_input)
print(f"\nFinal output shape: {output.shape}")

