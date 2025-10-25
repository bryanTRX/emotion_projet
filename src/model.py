import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv_layers = self._build_conv_layers()
        
        self.fc_layers = self._build_fc_layers()

    def _build_conv_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            
            nn.MaxPool2d(2,2)
        )
        
        return layers
    
    def _build_fc_layers(self):
        layers = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(128*12*12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 7)
        )
        
        return layers
        

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        return F.log_softmax(x, dim=1)