"""Window encoders and fixed or input-dependent mixtures of velocity heads."""

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, patch_size=50, patch_filters=32):
        super().__init__()
        if not isinstance(patch_size, int) or patch_size <= 0 or 200 % patch_size:
            raise ValueError("patch_size must be a positive divisor of the 200-sample input")
        if not isinstance(patch_filters, int) or patch_filters <= 0:
            raise ValueError("patch_filters must be a positive integer")
        self.patch_size = patch_size
        # A shared linear map on disjoint patches is a strided Conv1d.
        self.patch_conv = nn.Linear(patch_size * 6, patch_filters)
        self.gru = nn.GRU(patch_filters, 64, batch_first=True)

    def forward(self, x):
        patches = self.patch_conv(x.reshape(len(x), 200 // self.patch_size, self.patch_size * 6)).relu()
        _, last = self.gru(patches)
        return last[0]


class ConvTemporalEncoder(nn.Module):
    def __init__(self, kernel_size, stride, filters, use_gru=True):
        super().__init__()
        if (not isinstance(kernel_size, int) or not isinstance(stride, int)
                or not 1 <= stride <= kernel_size <= 200):
            raise ValueError("Expected 1 <= convolution stride <= kernel size <= 200")
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError("Convolution filter count must be a positive integer")
        self.right_padding = (kernel_size - 200) % stride
        self.patch_conv = nn.Conv1d(6, filters, kernel_size=kernel_size, stride=stride)
        self.gru = nn.GRU(filters, 64, batch_first=True) if use_gru else None
        if not use_gru:
            steps = 1 + (200 + self.right_padding - kernel_size) // stride
            input_size = steps * filters
            gru_parameters = 3 * 64 * (filters + 64 + 2)
            width = max(1, round((gru_parameters - 64) / (input_size + 65)))
            self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(input_size, width), nn.ReLU(), nn.Linear(width, 64))

    def forward(self, x):
        signal = x.transpose(1, 2)
        if self.right_padding:
            signal = nn.functional.pad(signal, (0, self.right_padding))
        features = self.patch_conv(signal).relu().transpose(1, 2)
        if self.gru is None:
            return self.mlp(features)
        _, last = self.gru(features)
        return last[0]


class WindowRegressor(nn.Module):
    def __init__(self, encoder, mixing, mean=None, std=None, patch_size=50, patch_filters=32, conv_stride=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(6) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(6) if std is None else torch.as_tensor(std, dtype=torch.float32))
        if encoder == "mlp":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(1200, 128), nn.ReLU(),
                                         nn.Linear(128, 64), nn.ReLU())
        elif encoder == "temporal":
            self.encoder = TemporalEncoder(patch_size, patch_filters)
        elif encoder in ("conv_temporal", "conv_mlp"):
            stride = patch_size if conv_stride is None else conv_stride
            self.encoder = ConvTemporalEncoder(patch_size, stride, patch_filters, use_gru=encoder == "conv_temporal")
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

    def forward(self, imu, *, return_gate_logits=False):
        if return_gate_logits and self.mixing != "gate":
            raise ValueError("Gate logits require adaptive mixing")
        if imu.ndim != 3 or imu.shape[1:] != (200, 6):
            raise ValueError(f"Expected (batch,200,6), got {tuple(imu.shape)}")
        z = self.encoder((imu - self.mean) / self.std)
        if self.mixing == "plain":
            return self.head(z)
        hidden = self.branch_features(z).relu().reshape(-1, 4, 16)
        branches = torch.einsum("bkh,khd->bkd", hidden, self.branch_weights) + self.branch_bias
        logits = self.mix_bias if self.mixing == "fixed" else self.gate(z) + self.mix_bias
        velocity = (branches * logits.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)
        return (velocity, logits) if return_gate_logits else velocity
