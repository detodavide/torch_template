import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class CustomResnet101(nn.Module):

    def __init__(self, num_classes):

        super(CustomResnet101).__init__()
        
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_in_features, num_classes)
        )

    def forward(self, x):

        x = self.resnet(x)
        return x