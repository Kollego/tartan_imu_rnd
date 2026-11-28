"""Train the HW2 generalist with platform supervision on its existing gate."""

import argparse
from pathlib import Path

from exps.hw2.study import ROOT, config_for
from exps.hw3.epoch_train import VARIANT
from exps.hw3.fixed_generalist import STEPS, train_generalist_fold
from src.data import PLATFORMS


GATE_WEIGHT = 0.1
RUN = ROOT / "gate_domain_ce_0.1_steps_20000_40000" / VARIANT


def run_fold(fold, output):
    config = config_for(VARIANT, "all")
    config["gate_domain_weight"] = GATE_WEIGHT
    train_generalist_fold(fold, output, config, Path(__file__), {"platforms_in_gate_order": PLATFORMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=config_for(VARIANT, "all")["folds"], required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    run_fold(args.fold, args.output)


if __name__ == "__main__":
    main()
