"""Train the Conv1d and GRU grid on the fixed HW2 generalist folds."""

import argparse
from pathlib import Path

from exps.hw2.study import ROOT, config_for
from exps.hw3.encoder_ablation import FILTERS, RUN as ENCODER_RUN
from exps.hw3.epoch_train import VARIANT
from exps.hw3.fixed_generalist import STEPS, train_generalist_fold


RUN = ROOT / "conv1d_gru_ablation_steps_20000_40000"
GEOMETRIES = {f"k{kernel}_s{stride}": (kernel, stride)
              for kernel in (50, 20, 10) for stride in (50, 20, 10) if kernel >= stride}
CONTROL = "k20_s20"
CONTROL_RUN = ENCODER_RUN / FILTERS


def convolution_config(geometry):
    config = config_for(VARIANT, "all")
    kernel, stride = GEOMETRIES[geometry]
    config["model_kwargs"].update(encoder="conv_temporal", patch_size=kernel, conv_stride=stride, patch_filters=64)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", choices=GEOMETRIES)
    parser.add_argument("--fold", type=int, choices=config_for(VARIANT, "all")["folds"], required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    config = convolution_config(args.geometry)
    train_generalist_fold(args.fold, args.output / args.geometry, config, Path(__file__))


if __name__ == "__main__":
    main()
