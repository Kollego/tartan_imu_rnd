from .mlp import MLP


def build_model(name, **kwargs):
    if name == "mlp":
        return MLP(**kwargs)
    raise ValueError(f"Unknown model: {name}")
