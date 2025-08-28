
import torch.nn as nn

class ShadowModel(nn.Module):
    """A 3-layer FCNN for the Shadow Model."""
    def __init__(self, input_dim, output_dim):
        super(ShadowModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class PrivateModel(nn.Module):
    """A 3-layer FCNN for the Private Model."""
    def __init__(self, input_dim, output_dim):
        super(PrivateModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class LocalHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LocalHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
