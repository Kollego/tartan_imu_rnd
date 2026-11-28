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
        for size, filters in ((25, 32), (50, 32), (100, 32), (20, 32), (20, 64)):
            with self.subTest(patch_size=size, patch_filters=filters):
                model = build_model("window", encoder="temporal", mixing="gate", patch_size=size,
                                    patch_filters=filters)
                patch = model.encoder.patch_conv
                expected = F.conv1d(x.transpose(1, 2), patch.weight.reshape(filters, size, 6).transpose(1, 2),
                                    patch.bias, stride=size).transpose(1, 2)
                captured = []
                hook = model.encoder.gru.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
                self.assertEqual(model(x).shape, (len(x), 3))
                hook.remove()
                torch.testing.assert_close(captured[0], expected.relu())

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

    def test_convolution_covers_the_last_samples_before_gru(self):
        x = torch.zeros(1, 200, 6)
        x[:, -1, :] = 1
        for kernel, stride, tokens in ((50, 50, 4), (50, 20, 9), (50, 10, 16),
                                       (20, 20, 10), (20, 10, 19), (10, 10, 20)):
            with self.subTest(kernel=kernel, stride=stride):
                model = build_model("window", encoder="conv_temporal", mixing="gate",
                                    patch_size=kernel, conv_stride=stride, patch_filters=64)
                conv = model.encoder.patch_conv
                with torch.no_grad():
                    conv.weight.fill_(1)
                    conv.bias.zero_()
                captured = []
                hook = model.encoder.gru.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
                self.assertEqual(model(x).shape, (1, 3))
                hook.remove()
                self.assertEqual(captured[0].shape, (1, tokens, 64))
                torch.testing.assert_close(captured[0][0, -1], torch.full((64,), 6.0))

    def test_conv_mlp_replaces_only_the_temporal_aggregation(self):
        common = dict(mixing="gate", patch_size=10, conv_stride=10, patch_filters=64)
        gru = build_model("window", encoder="conv_temporal", **common)
        mlp = build_model("window", encoder="conv_mlp", **common)
        self.assertEqual(gru.encoder.patch_conv.weight.shape, mlp.encoder.patch_conv.weight.shape)
        self.assertIsNone(mlp.encoder.gru)
        self.assertEqual(mlp.encoder.mlp[1].in_features, 20 * 64)
        self.assertEqual(mlp.encoder.mlp[1].out_features, 19)
        self.assertEqual(mlp(torch.randn(2, 200, 6)).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
