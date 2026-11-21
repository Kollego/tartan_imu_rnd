from .mlp import MLP
from .window import WindowRegressor


def build_model(name, **kwargs):
    if name == "mlp":
        return MLP(**kwargs)
    if name == "window":
        return WindowRegressor(**kwargs)
    raise ValueError(f"Unknown model: {name}")
