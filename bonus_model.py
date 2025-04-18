"""Define your architecture here."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = F.relu(self.depthwise(x))
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x

class BonusModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Depthwise separable blocks
        self.block1 = DepthwiseSeparableBlock(32, 64)
        self.block2 = DepthwiseSeparableBlock(64, 128)
        self.block3 = DepthwiseSeparableBlock(128, 256)
        self.conv3 = self.block3  # Added for Grad-CAM compatibility
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_bonus_model():
    """Return the bonus model."""
    model = BonusModel()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus.pt')['model'])
    return model