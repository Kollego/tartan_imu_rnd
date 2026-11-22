"""Run the whole paired matrix. Each worker trains and syncs one configuration."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product

from exps.hw2.results import is_archived
from exps.hw2.study import PATCH_SIZES, VARIANTS
from src.data import PLATFORMS


def run(pair):
    variant, platform = pair
    for module in (["track"] if is_archived(variant, platform) else ["train", "track"]):
        subprocess.run([sys.executable, "-m", f"exps.hw2.{module}", variant, platform], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--patch-study", action="store_true", help="Train and track the two alternate patch sizes")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        list(workers.map(run, product(PATCH_SIZES if args.patch_study else VARIANTS, ("all", *PLATFORMS))))
