"""Hold all models you wish to train."""
import torch
import torch.nn.functional as F

from torch import nn

from xcpetion import build_xception_backbone


class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


def get_xception_based_model() -> nn.Module:
    """Return an Xception-Based network.
    (1) Build an Xception pre-trained backbone using build_xception_backbone.
    (2) Override its fc attribute with the specified MLP head.
    """
    # A. Build Xception backbone
    custom_network = build_xception_backbone()
    
    # B. Override fc block with specified MLP architecture
    custom_network.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    return custom_network
