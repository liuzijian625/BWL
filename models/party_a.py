
import torch.nn as nn

class PartyAModel(nn.Module):
    """A 3-layer FCNN for Party A."""
    def __init__(self, input_dim, output_dim):
        super(PartyAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
