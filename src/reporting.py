"""Aggregate saved seeds into tables, uncertainty estimates, and plots."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import paired_bootstrap_delta
from .utils import save_json


def load_results(run_dir):
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "seed_*.json"))):
        with open(path) as handle:
            results.append(json.load(handle))
    return results


def _prediction_arrays(result):
    ordered = sorted(result["predictions"], key=lambda row: row["sentence_id"])
    targets = np.array([row["target"] for row in ordered])
    predictions = np.array([row["prediction"] for row in ordered])
    sentence_ids = np.array([row["sentence_id"] for row in ordered])
    return sentence_ids, targets, predictions


def build_summary(run_dir, bootstrap_samples=2000):
    results = load_results(run_dir)
    if not results:
        raise FileNotFoundError(f"no completed results under {run_dir}")
    grouped = {}
    for result in results:
        grouped.setdefault(result["setup"], []).append(result)

    rows = []
    for setup, values in sorted(grouped.items()):
        accuracies = [value["oof"]["accuracy"] for value in values]
        f1s = [value["oof"]["macro_f1"] for value in values]
        gate_values = [
            fold["gate_mean"]
            for value in values
            for fold in value["folds"]
            if fold.get("gate_mean") is not None
        ]
        row = {
            "setup": setup,
            "n_seeds": len(values),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "gate_mean": float(np.mean(gate_values)) if gate_values else np.nan,
        }

        baseline_name = None
        if setup != "eeg_only" and not setup.startswith("text_"):
            baseline_name = "text_finetune" if setup.endswith("finetune") else "text_frozen"
        comparisons = []
        if baseline_name in grouped:
            baseline_by_seed = {value["seed"]: value for value in grouped[baseline_name]}
            for candidate in values:
                baseline = baseline_by_seed.get(candidate["seed"])
                if baseline is None:
                    continue
                ids_base, targets_base, predictions_base = _prediction_arrays(baseline)
                ids_cand, targets_cand, predictions_cand = _prediction_arrays(candidate)
                if not np.array_equal(ids_base, ids_cand) or not np.array_equal(
                    targets_base, targets_cand
                ):
                    raise ValueError("paired results do not contain the same sentences")
                comparisons.append(
                    paired_bootstrap_delta(
                        targets_cand,
                        predictions_base,
                        predictions_cand,
                        seed=candidate["seed"],
                        samples=bootstrap_samples,
                    )
                )
        if comparisons:
            row["baseline"] = baseline_name
            row["delta_macro_f1"] = float(
                np.mean([item["delta_macro_f1"] for item in comparisons])
            )
            row["delta_ci95_low"] = float(
                np.mean([item["ci95_low"] for item in comparisons])
            )
            row["delta_ci95_high"] = float(
                np.mean([item["ci95_high"] for item in comparisons])
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    tables_dir = os.path.join(run_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    frame.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)
    save_json(rows, os.path.join(tables_dir, "summary.json"))
    _write_markdown(frame, os.path.join(tables_dir, "summary.md"))
    _plot_scores(frame, os.path.join(run_dir, "plots", "scores.png"))
    _plot_confusions(grouped, os.path.join(run_dir, "plots", "confusions.png"))
    return frame


def _write_markdown(frame, path):
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(4)
    columns = list(display.columns)
    lines = [
        "# ZuCo multimodal sentiment results",
        "",
        "Macro-F1 is the primary metric. Delta intervals are paired sentence-level ",
        "bootstrap intervals against the matching text-only setup.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _plot_scores(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.55 * len(frame))))
    for ax, metric, label in [
        (axes[0], "accuracy", "accuracy"),
        (axes[1], "macro_f1", "macro-F1"),
    ]:
        ax.barh(
            y,
            frame[f"{metric}_mean"],
            xerr=frame[f"{metric}_std"],
            color="#4C78A8",
            capsize=3,
        )
        ax.set_yticks(y, frame["setup"])
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Out-of-fold sentence classification")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def _plot_confusions(grouped, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    setups = sorted(grouped)
    ncols = 3
    nrows = int(np.ceil(len(setups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, setup in zip(axes, setups):
        matrices = [np.asarray(value["oof"]["confusion_matrix"]) for value in grouped[setup]]
        matrix = np.mean(matrices, axis=0)
        normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        names = grouped[setup][0]["oof"]["class_names"]
        ax.set_xticks(range(3), names, rotation=30, ha="right")
        ax.set_yticks(range(3), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(setup)
        for row in range(3):
            for column in range(3):
                color = "white" if normalized[row, column] > 0.5 else "black"
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                )
    for ax in axes[len(setups) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
