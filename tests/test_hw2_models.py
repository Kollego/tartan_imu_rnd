"""Verify the ablation controls, patch convention and checkpoint interface."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from src.inference import load_model
from src.model import build_model


class HW2Models(unittest.TestCase):
    def test_patch_operation_matches_strided_convolution(self):
        x = torch.randn(3, 200, 6)
        patch = build_model("window", encoder="temporal", mixing="plain").encoder.patch_conv
        expected = F.conv1d(x.transpose(1, 2), patch.weight.reshape(32, 50, 6).transpose(1, 2),
                            patch.bias, stride=50).transpose(1, 2)
        torch.testing.assert_close(patch(x.reshape(len(x), 4, 300)), expected)

    def test_gate_starts_at_fixed_mixture_for_both_encoders(self):
        x = torch.randn(3, 200, 6)
        for encoder in ("mlp", "temporal"):
            torch.manual_seed(42)
            fixed = build_model("window", encoder=encoder, mixing="fixed")
            torch.manual_seed(42)
            gated = build_model("window", encoder=encoder, mixing="gate")
            for key, tensor in fixed.state_dict().items():
                torch.testing.assert_close(tensor, gated.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(fixed(x), gated(x))
            gated(x).square().mean().backward()
            self.assertGreater(gated.gate.weight.grad.abs().sum().item(), 0)
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "final.pt"
                torch.save({"model_name": "window", "model_kwargs": dict(encoder=encoder, mixing="gate"),
                            "state_dict": gated.state_dict()}, path)
                restored, _ = load_model(path)
                torch.testing.assert_close(restored(x), gated(x))


if __name__ == "__main__":
    unittest.main()
