import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)    # 3×3 pattern detection
        self.pool1 = nn.MaxPool2d(2, 2)     # 2×2 compression

        self.conv2 = nn.Conv2d(32, 64, 3)   # taking our 32 maps and detecting 64 further features
        self.pool2 = nn.MaxPool2d(2, 2)     # 2×2 compression

        self.fc = nn.Linear(2304, 10)       # Final classification, 2304 inputs, 10 outputs
        
    def forward(self, x):
        x = self.conv1(x)    # Apply first conv layer
        x = self.pool1(x)    # Apply first pooling 
        x = self.conv2(x)    # Apply second conv layer
        x = self.pool2(x)    # Apply second pooling
        x = torch.flatten(x, 1)  # Flatten, need to say on 1st dimension so we only have a 0th dimmension of linear values
        x = self.fc(x)       # Final classification
        return x             