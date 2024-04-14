import torch
import torch.nn as nn

class ScaleNet(nn.Module):

    def __init__(self, feature_extractor: nn.Module) -> None:
        super(ScaleNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.l1 = nn.Linear(512, 256)  # Change 512 to the number of features extracted by the feature extractor
        self.r = nn.ReLU()
        self.l2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        x = self.l1(features)
        x = self.r(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x.flatten()