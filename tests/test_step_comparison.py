"""Fixed-budget reports must use the requested snapshot and real batch counters."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.compare_steps import align_oof, compare, read_run, read_snapshot
from src.evaluation.plot_comparison import learning_curves, plot_comparison
from src.data import PLATFORMS


class StepComparisonTests(unittest.TestCase):
    def make_runs(self, root, steps=(2000, 8000)):
        rows = []
        for platform in PLATFORMS:
            for fold in range(5):
                for window in range(100):
                    rows.append({"window_id": len(rows), "platform": platform,
                                 "traj_id": f"{platform}_{fold}", "win_idx": window,
                                 "fold": fold, "vx": 1., "vy": 0., "vz": 0.})
        all_rows = pd.DataFrame(rows)
        folds = all_rows[["traj_id", "platform", "fold"]].drop_duplicates()
        directories = [root / "generalist", *[root / p for p in PLATFORMS]]
        for directory, platform in zip(directories, [None, *PLATFORMS], strict=True):
            directory.mkdir()
            config = {"platform": platform, "output_dir": str(directory), "max_steps": steps[-1],
                      "snapshot_steps": list(steps), "eval_every_steps": 250,
                      "batch_size": 256, "drop_last": True, "n_splits": 5, "folds": list(range(5))}
            (directory / "config.json").write_text(json.dumps(config))
            (directory / "environment.json").write_text(json.dumps({"index_sha256": {"train": "same"}}))
            (directory / "run_compute.json").write_text(json.dumps({"loading_seconds": 1.}))
            folds.to_csv(directory / "folds.csv", index=False)
            selected = all_rows if platform is None else all_rows[all_rows.platform.eq(platform)]
            for step in steps:
                snapshot = directory / f"step_{step}"
                snapshot.mkdir()
                frame = selected.copy()
                # Different predictions catch accidental use of the wrong snapshot.
                frame["pred_vx"] = 0. if platform is None else (1. if step == steps[0] else 0.5)
                frame["pred_vy"], frame["pred_vz"] = 0., 0.
                frame.sample(frac=1, random_state=1).to_csv(snapshot / "oof_predictions.csv", index=False)
                metrics = pd.DataFrame([
                    {"fold": fold, "step": step, "train_steps": step, "windows_seen": step * 256,
                     "train_windows": int(selected.fold.ne(fold).sum()),
                     "train_seconds": 2., "train_cpu_seconds": 4.,
                     "evaluation_seconds": 0.5, "evaluation_cpu_seconds": 1.,
                     "normalization_seconds": 0.1, "normalization_cpu_seconds": 0.2,
                     "elapsed_cpu_seconds": 6.,
                     "validation_calls": step // 250, "elapsed_seconds": 3.}
                    for fold in range(5)])
                metrics.to_csv(snapshot / "fold_metrics.csv", index=False)
        return directories

    def test_10k_5k_uses_half_snapshot_and_twice_the_group_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.make_runs(root, steps=(5000, 10000))
            output = root / "comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                compare(directories[0], directories[1:], output)
            full = json.loads((output / "per_model_10000" / "summary.json").read_text())
            short = json.loads((output / "specialists_5000" / "summary.json").read_text())
            self.assertEqual(short["generalist"]["oof_macro_ave"], 1.)
            self.assertEqual(short["specialists"]["oof_macro_ave"], 0.)
            self.assertEqual(full["specialists"]["oof_macro_ave"], 0.5)
            self.assertEqual(short["specialist_group_update_ratio"], 2.)
            self.assertEqual(full["specialist_group_update_ratio"], 4.)
            self.assertEqual(short["generalist_steps_per_fold"], 10000)
            self.assertEqual(short["specialist_steps_each_per_fold"], 5000)
            totals = short["measured_compute_totals"]
            self.assertEqual(totals["specialists"]["train_steps_across_folds"],
                             2 * totals["generalist"]["train_steps_across_folds"])

            # Unequal platform means must receive equal weights within each fold.
            for domain, directory in enumerate(directories):
                pd.DataFrame([{"fold": f, "step": s, "cv_macro_ave": domain + f / 10 + s / 10000}
                              for f in range(5) for s in (5000, 10000)]).to_csv(
                                  directory / "history.csv", index=False)
            curves = learning_curves(directories[0], directories[1:])
            self.assertAlmostEqual(curves.generalist_mean.iloc[0], 0.7)
            self.assertAlmostEqual(curves.specialists_mean.iloc[0], 3.2)
            self.assertAlmostEqual(curves.specialists_std.iloc[0], np.std(np.arange(5) / 10, ddof=1))
            plot_comparison(directories[0], directories[1:], output)
            self.assertTrue((output / "comparison.png").exists())
            self.assertEqual(len(pd.read_csv(output / "learning_curves.csv")), 2)
            path = directories[1] / "history.csv"
            pd.read_csv(path).iloc[1:].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                learning_curves(directories[0], directories[1:])

    def test_scenarios_use_final_requested_snapshots_and_distinct_compute_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.make_runs(root)
            output = root / "comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                compare(directories[0], directories[1:], output)
            per_model = json.loads((output / "per_model_8000" / "summary.json").read_text())
            total = json.loads((output / "total_8000" / "summary.json").read_text())
            self.assertEqual(per_model["generalist"]["oof_macro_ave"], 1.)
            self.assertEqual(per_model["specialists"]["oof_macro_ave"], 0.5)
            self.assertEqual(total["specialists"]["oof_macro_ave"], 0.)
            p = per_model["measured_compute_totals"]
            t = total["measured_compute_totals"]
            self.assertEqual(p["specialists"]["train_steps_across_folds"], 4 * p["generalist"]["train_steps_across_folds"])
            self.assertEqual(t["specialists"]["windows_seen_across_folds"], t["generalist"]["windows_seen_across_folds"])
            self.assertEqual(t["specialists"]["evaluation_seconds"], 10.)
            self.assertEqual(len(pd.read_csv(output / "total_8000" / "budget.csv")), 25)
            with self.assertRaises(FileExistsError):
                compare(directories[0], directories[1:], output)

    def test_invalid_budget_counters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = self.make_runs(Path(tmp))
            run = read_run(directories[0])
            path = directories[0] / "step_8000" / "fold_metrics.csv"
            original = pd.read_csv(path)
            for column, value in (("train_steps", 7999), ("windows_seen", 8000 * 256 - 1),
                                  ("step", 2000), ("train_windows", 1), ("train_seconds", np.nan)):
                modified = original.copy()
                modified.loc[0, column] = value
                modified.to_csv(path, index=False)
                with self.subTest(column=column), self.assertRaises(ValueError):
                    read_snapshot(run, 8000)


class ReferenceBudgetTests(unittest.TestCase):
    def make_reference_runs(self, root):
        directories = StepComparisonTests.make_runs(self, root)
        # Unequal trajectory lengths produce fold-dependent, non-divisible budgets.
        base = pd.read_csv(directories[0] / "step_8000" / "oof_predictions.csv")
        base = base[base.win_idx < (100 - base.fold * 10)].copy()
        windows = pd.Series({fold: int(base.fold.ne(fold).sum()) for fold in range(5)})
        steps = 3 * (windows // 256)
        self.assertGreater(steps.nunique(), 1)
        self.assertTrue((steps % 4 != 0).any())
        for directory, platform in zip(directories, [None, *PLATFORMS], strict=True):
            config = json.loads((directory / "config.json").read_text())
            del config["max_steps"], config["snapshot_steps"]
            config["reference_epochs"] = 3
            (directory / "config.json").write_text(json.dumps(config))
            short = steps if platform is None else pd.Series({
                fold: int(steps[fold] // 4 + ((PLATFORMS.index(platform) - fold) % 4 < steps[fold] % 4))
                for fold in range(5)})
            pd.DataFrame({"fold": range(5), "generalist_train_windows": windows,
                          "generalist_steps": steps, "equal_total_steps": short}).to_csv(
                              directory / "fold_budgets.csv", index=False)
            selected = base if platform is None else base[base.platform.eq(platform)]
            for name in (["final"] if platform is None else ["final", "equal_total"]):
                snapshot = directory / name
                snapshot.mkdir()
                frame = selected.copy()
                frame["pred_vx"] = 0. if platform is None else (0.5 if name == "final" else 1.)
                frame.to_csv(snapshot / "oof_predictions.csv", index=False)
                metrics = pd.read_csv(directory / "step_8000" / "fold_metrics.csv")
                budget = steps if name == "final" else short
                metrics["step"] = metrics.fold.map(budget)
                metrics["train_steps"] = metrics.step
                metrics["windows_seen"] = metrics.step * 256
                metrics["train_windows"] = metrics.fold.map(lambda fold: int(selected.fold.ne(fold).sum()))
                metrics["validation_calls"] = 1
                metrics.to_csv(snapshot / "fold_metrics.csv", index=False)
        return directories, steps

    def test_reference_passes_keep_exact_per_fold_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories, steps = self.make_reference_runs(Path(tmp))
            output = Path(tmp) / "comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                compare(directories[0], directories[1:], output)
            per_model = json.loads((output / "per_model_3passes" / "summary.json").read_text())
            total = json.loads((output / "total_3passes" / "summary.json").read_text())
            self.assertEqual(per_model["generalist_steps_by_fold"], {str(f): int(s) for f, s in steps.items()})
            self.assertEqual(per_model["specialists"]["oof_macro_ave"], 0.5)
            self.assertEqual(total["specialists"]["oof_macro_ave"], 0.)
            self.assertEqual(total["reference_epochs"], 3)
            for fold, count in steps.items():
                quotas = total["specialist_steps_by_fold"][str(fold)]
                self.assertEqual(sum(quotas.values()), count)
                self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)
                for platform in PLATFORMS:
                    self.assertEqual(per_model["specialist_steps_by_fold"][str(fold)][platform], count)
                    expected = count // 4 + int((PLATFORMS.index(platform) - fold) % 4 < count % 4)
                    self.assertEqual(quotas[platform], expected)
            totals = total["measured_compute_totals"]
            self.assertEqual(totals["generalist"]["windows_seen_across_folds"],
                             totals["specialists"]["windows_seen_across_folds"])

    def test_reference_budget_and_snapshot_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories, steps = self.make_reference_runs(Path(tmp))
            specialist = directories[1]
            path = specialist / "fold_budgets.csv"
            budgets = pd.read_csv(path)
            modified = budgets.copy()
            modified.loc[0, "equal_total_steps"] += 1
            modified.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_run(specialist)
            budgets.to_csv(path, index=False)
            run = read_run(specialist)
            metrics_path = specialist / "equal_total" / "fold_metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metrics["step"] = metrics.fold.map(steps)
            metrics["train_steps"] = metrics.step
            metrics["windows_seen"] = metrics.step * 256
            metrics.to_csv(metrics_path, index=False)
            with self.assertRaises(ValueError):
                read_snapshot(run, "equal_total")
            with self.assertRaises(ValueError):
                read_snapshot(read_run(directories[0]), "equal_total")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"window_id": [10, 11], "platform": ["car", "car"],
                                   "traj_id": ["a", "b"], "win_idx": [0, 0], "fold": [0, 1],
                                   "vx": [1., 2.], "vy": [0., 0.], "vz": [0., 0.],
                                   "pred_vx": [3., 4.], "pred_vy": [0., 0.], "pred_vz": [0., 0.]})

    def test_pairing_uses_ids_and_rejects_mismatched_windows(self):
        _, aligned = align_oof(self.frame, self.frame.iloc[::-1])
        self.assertEqual(aligned.window_id.tolist(), [10, 11])
        self.assertEqual(aligned.pred_vx.tolist(), [3., 4.])
        changed_target = self.frame.copy()
        changed_target.loc[0, "vx"] = 100
        changed_fold = self.frame.copy()
        changed_fold.loc[0, "fold"] = 4
        for other in (self.frame.iloc[:1], pd.concat([self.frame, self.frame.iloc[:1]]), changed_target, changed_fold):
            with self.subTest(), self.assertRaises(ValueError):
                align_oof(self.frame, other)


if __name__ == "__main__":
    unittest.main()
