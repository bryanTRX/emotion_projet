import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()
        self.num_classes = num_classes
        self.conv_layers = self._build_conv_layers()
        self.fc_layers = self._build_fc_layers()

    def _build_conv_layers(self):
        layers = nn.Sequential(
            # Bloc 1 
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            
            # Bloc 2 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),

            # Bloc 3 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.35),

            # Bloc 4 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),
            # nn.AdaptiveAvgPool2d(1)
        )
        
        return layers
    
    def _build_fc_layers(self):
        layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, self.num_classes)
        )
        
        return layers
        

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        return x