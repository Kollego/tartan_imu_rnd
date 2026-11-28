"""Train the two successive temporal-encoder ablations against the HW2 generalist."""

import argparse
from pathlib import Path

from exps.hw2.study import ROOT, config_for
from exps.hw3.epoch_train import VARIANT, output_for
from exps.hw3.fixed_generalist import STEPS, train_generalist_fold


RUN = ROOT / "encoder_ablation_steps_20000_40000"
GEOMETRIES = {"p25_f32": (25, 32), "p20_f32": (20, 32), "p20_f64": (20, 64)}
BASELINE, PATCHES, FILTERS = GEOMETRIES
BASELINE_RUN = output_for(steps=STEPS) / "all"


def ablation_config(geometry):
    config = config_for(VARIANT, "all")
    patch_size, patch_filters = GEOMETRIES[geometry]
    config["model_kwargs"].update(patch_size=patch_size, patch_filters=patch_filters)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", choices=(PATCHES, FILTERS))
    parser.add_argument("--fold", type=int, choices=config_for(VARIANT, "all")["folds"], required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    config = ablation_config(args.geometry)
    train_generalist_fold(args.fold, args.output / args.geometry, config, Path(__file__))


if __name__ == "__main__":
    main()
