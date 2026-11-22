"""Paired ablations, domain diagnostics and publication figures for HW2."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exps.hw2.results import OUT

CONTRASTS = {
    "temporal_vs_mlp_plain": ("mlp_plain", "temporal_plain"),
    "temporal_vs_mlp_gated": ("mlp_gate", "temporal_gate"),
    "gate_vs_fixed_temporal": ("temporal_fixed", "temporal_gate"),
    "gate_vs_fixed_mlp": ("mlp_fixed", "mlp_gate"),
    "heads_vs_plain_mlp": ("mlp_plain", "mlp_fixed"),
    "heads_vs_plain_temporal": ("temporal_plain", "temporal_fixed"),
}
COLORS = {"generalist": "#0066a0", "specialists": "#ba572b"}
LABELS = {"generalist": "Генералист", "specialists": "Специалисты"}
NAMES = {"plain": "Одна голова", "fixed": "Общие веса ветвей", "gate": "Gate по IMU"}


def main():
    folds = pd.read_csv(OUT / "folds.csv").pivot(index="fold", columns=["variant", "role"], values="ave")
    rows = []
    for name, (control, changed) in CONTRASTS.items():
        gains = folds[control] - folds[changed]
        for fold, row in gains.iterrows():
            rows.append({"contrast": name, "control": control, "changed": changed, "fold": fold,
                         "gain_generalist": row.generalist, "gain_specialists": row.specialists,
                         "extra_gain_generalist": row.generalist - row.specialists})
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT / "paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, layout="constrained")
    for ax, contrast, title in zip(axes, ["temporal_vs_mlp_gated", "gate_vs_fixed_temporal"],
                                   ["Замена MLP на Conv1d + GRU\nпри включённом gate",
                                    "Включение gate по IMU\nпри Conv1d + GRU"]):
        changes = paired[paired.contrast.eq(contrast)]
        ax.axhline(0, color="#425361", linewidth=1)
        for i, role in enumerate(COLORS):
            y = changes[f"gain_{role}"]
            x = changes.fold + 1 + (i - 0.5) * 0.12
            ax.scatter(x, y, s=38, label=LABELS[role], marker=["o", "s"][i], color=COLORS[role])
            ax.axhline(y.mean(), linestyle="--", color=COLORS[role], linewidth=1)
        ax.set_title(title)
        ax.set_xticks(range(1, 6), range(5))
        ax.set_xlabel("Фолд")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Снижение macro AVE, м/с")
    axes[1].legend(frameon=False)
    fig.suptitle("Парные эффекты двух изменений  ·  пунктир показывает среднее", fontsize=13)
    fig.savefig(OUT / "ablations.png", dpi=170)
    plt.close(fig)

    curves = pd.read_csv(OUT / "curves.csv").groupby(["role", "variant", "step"]).cv_macro_ave.mean()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True, sharey=True, layout="constrained")
    for row, role in enumerate(COLORS):
        for col, arch in enumerate(["mlp", "temporal"]):
            ax = axes[row, col]
            for mix, color in zip(NAMES, ["#405869", "#b75521", "#00877b"]):
                values = curves.loc[role, f"{arch}_{mix}"]
                ax.plot(values.index, values, label=NAMES[mix], color=color, lw=1.6)
            ax.set_title(f"{LABELS[role]} · " + ("MLP" if arch == "mlp" else "Conv1d + GRU"))
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if not col:
                ax.set_ylabel("CV macro AVE, м/с")
            if row:
                ax.set_xlabel("Шаг оптимизатора")
    axes[0, 1].legend(frameon=False)
    fig.savefig(OUT / "learning.png", dpi=170)
    plt.close(fig)


    domains = pd.read_csv(OUT / "domains.csv").pivot(
        index="platform", columns=["variant", "role"], values="ave")
    fig, ax = plt.subplots(figsize=(8.6, 3.5), layout="constrained")
    ax.hlines(range(len(domains)), 0, 1, color="#dedede", linewidth=0.8, zorder=0)
    for variant, role, label, marker, color in [
        ("mlp_plain", "generalist", "MLP · генералист", "o", "#687783"),
        ("temporal_gate", "generalist", "GRU + gate · генералист", "D", COLORS["generalist"]),
        ("temporal_gate", "specialists", "GRU + gate · специалисты", "s", COLORS["specialists"]),
    ]:
        ax.scatter(domains[variant, role], domains.index, label=label, marker=marker, color=color, s=52)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.margins(y=0.22)
    ax.set_xlabel("OOF AVE по платформе, м/с  ·  меньше лучше")
    ax.set_title("У дрона остаётся самая высокая ошибка")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(OUT / "platform_errors.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
