"""XXX
"""
from torch import nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    def __init__(
        self, *, fc1_layers: int = 1024, fc2_layers: int = 128, dropout_p: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, fc1_layers)
        self.fc2 = nn.Linear(fc1_layers, fc2_layers)
        self.fc3 = nn.Linear(fc2_layers, 10)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
