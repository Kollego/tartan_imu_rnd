"""Small fully connected baseline: one second of IMU -> mean velocity."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(128, 64), mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(6) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(6) if std is None else torch.as_tensor(std, dtype=torch.float32))
        layers = [nn.Flatten()]
        width = 200 * 6
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, imu):
        if imu.ndim != 3 or imu.shape[1:] != (200, 6):
            raise ValueError(f"Expected (batch,200,6), got {tuple(imu.shape)}")
        return self.network((imu - self.mean) / self.std)
