"""Fixed HW2 protocol. Each configuration uses the identical trajectory folds."""

from pathlib import Path

from exps.baseline.configs.mlp_steps import CONFIG

ROOT = Path("runs/hw2")
PROJECT = "TartanIMU/HW2"
FOLDS = Path(__file__).with_name("folds.csv")
# Absolute change in specialist mean CV macro AVE deemed practically small.
MARGIN = 0.01
VARIANTS = tuple(f"{encoder}_{mix}" for encoder in ("mlp", "temporal")
                 for mix in ("plain", "fixed", "gate"))
HYPOTHESES = {
    "plain": "The temporal encoder can exploit within-window dynamics.",
    "fixed": "Multiple nonlinear heads with global weights control for branch capacity.",
    "gate": "Input-dependent head weights help joint training more than single-domain training.",
}


def config_for(variant, platform):
    encoder, mixing = variant.split("_")
    config = dict(CONFIG, platform=None if platform == "all" else platform,
                  snapshot_steps=[10000], output_dir=str(ROOT / variant / platform))
    if variant != "mlp_plain":
        config.update(model_name="window", model_kwargs={"encoder": encoder, "mixing": mixing})
    return config
