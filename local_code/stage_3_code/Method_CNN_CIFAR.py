"""CIFAR CNN architecture used by script_cnn_cifar.py."""
from torch import nn


class CIFARCNN(nn.Module):
    def __init__(self, depth=3, batchnorm=True, dropout=0.3):
        super().__init__()
        layers, incoming = [], 3
        for outgoing in (32, 64, 128)[:depth]:
            for _ in range(2):
                layers.append(nn.Conv2d(incoming, outgoing, 3, padding=1))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(outgoing))
                layers.append(nn.ReLU())
                incoming = outgoing
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(incoming * (32 // 2**depth)**2, 128),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 10))
    def forward(self, x):
        return self.head(self.features(x))
