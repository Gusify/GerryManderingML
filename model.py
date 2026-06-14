import torch.nn as nn

NUM_FEATURES = 4


class VoteMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
