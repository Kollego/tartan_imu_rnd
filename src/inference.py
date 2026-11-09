"""Common inference: raw (batch,200,6) -> body-frame velocity (batch,3)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import build_model


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, dataset, batch_size=512, device="cpu"):
    predictions = []
    for imu, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        predictions.append(model(imu.to(device)).cpu().numpy())
    result = np.concatenate(predictions) if predictions else np.empty((0, 3), dtype=np.float32)
    if result.shape != (len(dataset), 3) or not np.isfinite(result).all():
        raise ValueError("Invalid predictions")
    return result
