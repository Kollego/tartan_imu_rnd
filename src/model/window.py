"""Window encoders and fixed or input-dependent mixtures of velocity heads."""

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Four ordered 250-ms patches. A shared linear map on disjoint patches
        # is a Conv1d with kernel size and stride 50, in time-major layout.
        self.patch_conv = nn.Linear(50 * 6, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)

    def forward(self, x):
        patches = self.patch_conv(x.reshape(len(x), 4, 300)).relu()
        _, last = self.gru(patches)
        return last[0]


class WindowRegressor(nn.Module):
    def __init__(self, encoder, mixing, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(6) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(6) if std is None else torch.as_tensor(std, dtype=torch.float32))
        if encoder == "mlp":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(1200, 128), nn.ReLU(),
                                         nn.Linear(128, 64), nn.ReLU())
        elif encoder == "temporal":
            self.encoder = TemporalEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        if mixing not in ("plain", "fixed", "gate"):
            raise ValueError(f"Unknown mixing: {mixing}")
        self.mixing = mixing
        if mixing == "plain":
            self.head = nn.Linear(64, 3)
        else:
            # Each of four branches gets 16 private channels and a 3D output.
            self.branch_features = nn.Linear(64, 4 * 16)
            self.branch_weights = nn.Parameter(torch.empty(4, 16, 3))
            self.branch_bias = nn.Parameter(torch.zeros(4, 3))
            nn.init.uniform_(self.branch_weights, -0.25, 0.25)
            self.mix_bias = nn.Parameter(torch.zeros(4))
            if mixing == "gate":
                self.gate = nn.Linear(64, 4, bias=False)
                nn.init.zeros_(self.gate.weight)

    def forward(self, imu):
        if imu.ndim != 3 or imu.shape[1:] != (200, 6):
            raise ValueError(f"Expected (batch,200,6), got {tuple(imu.shape)}")
        z = self.encoder((imu - self.mean) / self.std)
        if self.mixing == "plain":
            return self.head(z)
        hidden = self.branch_features(z).relu().reshape(-1, 4, 16)
        branches = torch.einsum("bkh,khd->bkd", hidden, self.branch_weights) + self.branch_bias
        logits = self.mix_bias if self.mixing == "fixed" else self.gate(z) + self.mix_bias
        return (branches * logits.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)
