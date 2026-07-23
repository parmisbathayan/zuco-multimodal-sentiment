"""Post-hoc analysis of saved gated-control predictions without retraining."""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import CLASS_NAMES, LABEL_TO_ID
from .data import load_multimodal_data
from .labels import load_labels
from .reporting import load_results
from .utils import load_json, save_json


TEXT_MODES = ("finetune", "frozen")
CONTROLS = ("shuffled", "noise", "zero")
PRIORITY_SUBSETS = (
    "low_confidence_25",
    "low_confidence_50",
    "no_eeg_incorrect",
)


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _word_count(sentence):
    return len(re.findall(r"\b\w+\b", str(sentence)))


def _result_index(run_dir):
    indexed = {}
    for result in load_results(run_dir):
        key = (result["setup"], int(result["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate result for setup/seed {key}")
        indexed[key] = result
    return indexed


def _required_results(indexed):
    missing = []
    for text_mode in TEXT_MODES:
        setups = [f"gated_{text_mode}"] + [
            f"gated_{control}_{text_mode}" for control in CONTROLS
        ]
        seed_sets = []
        for setup in setups:
            seeds = sorted(seed for name, seed in indexed if name == setup)
            if not seeds:
                missing.append(setup)
            else:
                seed_sets.append((setup, seeds))
        if seed_sets and len({tuple(seeds) for _, seeds in seed_sets}) != 1:
            raise ValueError(
                f"controlled setups do not contain matching seeds for {text_mode}: "
                f"{seed_sets}"
            )
    if missing:
        raise ValueError(
            "the source run is missing controlled results: " + ", ".join(missing)
        )


def _sentence_metadata(labels_csv, features_dir):
    labels = load_labels(labels_csv)
    metadata = labels[["sentence_id", "sentence", "sentiment_label"]].copy()
    metadata["target"] = metadata["sentiment_label"].map(LABEL_TO_ID)
    if metadata["target"].isna().any():
        raise ValueError("sentiment labels must be -1, 0, or 1")
    metadata["target"] = metadata["target"].astype(int)
    metadata["word_count"] = metadata["sentence"].map(_word_count)

    data = load_multimodal_data(labels_csv, features_dir)
    reader_counts = pd.DataFrame(
        {
            "sentence_id": data.sentence_ids.astype(int),
            "reader_count": data.subject_mask.sum(axis=1).astype(int),
        }
    )
    metadata = metadata.merge(reader_counts, on="sentence_id", validate="one_to_one")
    low = float(metadata["word_count"].quantile(0.25))
    high = float(metadata["word_count"].quantile(0.75))
    metadata["length_group"] = np.select(
        [metadata["word_count"] <= low, metadata["word_count"] > high],
        ["short", "long"],
        default="medium",
    )
    return metadata


def build_sentence_diagnostics(indexed, metadata):
    """Create one held-out diagnostic row per setup, seed, and sentence."""
    metadata_by_id = metadata.set_index("sentence_id").to_dict(orient="index")
    rows = []
    controlled_setups = sorted(
        {setup for setup, _ in indexed if setup.startswith("gated_")}
    )
    for setup in controlled_setups:
        for (name, seed), result in sorted(indexed.items()):
            if name != setup:
                continue
            for prediction in result["predictions"]:
                sentence_id = int(prediction["sentence_id"])
                meta = metadata_by_id.get(sentence_id)
                if meta is None:
                    raise ValueError(f"sentence {sentence_id} is absent from labels CSV")
                target = int(prediction["target"])
                if target != int(meta["target"]):
                    raise ValueError(f"target mismatch for sentence {sentence_id}")
                diagnostics = prediction.get("diagnostics")
                if not diagnostics:
                    raise ValueError(
                        f"{setup}, seed {seed}, sentence {sentence_id} has no diagnostics"
                    )
                with_logits = np.asarray(diagnostics["logits_with_eeg"], dtype=float)
                without_logits = np.asarray(
                    diagnostics["logits_without_eeg"], dtype=float
                )
                with_prediction = int(with_logits.argmax())
                saved_prediction = int(prediction["prediction"])
                if with_prediction != saved_prediction:
                    raise ValueError(
                        f"saved prediction/logit mismatch for {setup}, seed {seed}, "
                        f"sentence {sentence_id}"
                    )
                without_probabilities = _softmax(without_logits)
                ranked = np.sort(without_probabilities)
                without_prediction = int(without_logits.argmax())
                with_correct = with_prediction == target
                without_correct = without_prediction == target
                logit_delta = with_logits - without_logits
                text_norm = float(diagnostics["text_embedding_norm"])
                contribution_norm = float(
                    diagnostics["gated_eeg_contribution_norm"]
                )
                rows.append(
                    {
                        "setup": setup,
                        "text_mode": result["text_mode"],
                        "eeg_control": result["eeg_control"],
                        "seed": int(seed),
                        "sentence_id": sentence_id,
                        "sentence": meta["sentence"],
                        "target": target,
                        "class_name": CLASS_NAMES[target],
                        "word_count": int(meta["word_count"]),
                        "length_group": meta["length_group"],
                        "reader_count": int(meta["reader_count"]),
                        "prediction_with_eeg": with_prediction,
                        "prediction_without_eeg": without_prediction,
                        "correct_with_eeg": int(with_correct),
                        "correct_without_eeg": int(without_correct),
                        "prediction_changed": int(
                            with_prediction != without_prediction
                        ),
                        "favorable_flip": int(with_correct and not without_correct),
                        "unfavorable_flip": int(
                            without_correct and not with_correct
                        ),
                        "text_path_confidence": float(
                            without_probabilities.max()
                        ),
                        "text_path_margin": float(ranked[-1] - ranked[-2]),
                        "text_embedding_norm": text_norm,
                        "eeg_embedding_norm": float(
                            diagnostics["eeg_embedding_norm"]
                        ),
                        "candidate_eeg_contribution_norm": float(
                            diagnostics["candidate_eeg_contribution_norm"]
                        ),
                        "gated_eeg_contribution_norm": contribution_norm,
                        "contribution_to_text_norm_ratio": contribution_norm
                        / max(text_norm, 1e-12),
                        "logit_delta_l2": float(np.linalg.norm(logit_delta)),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no gated sentence diagnostics were found")
    return frame


def _confidence_quartiles(frame):
    frame = frame.copy()
    frame["confidence_quartile"] = (
        frame.groupby(["text_mode", "eeg_control", "seed"])[
            "text_path_confidence"
        ]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=["q1_lowest", "q2", "q3", "q4_highest"],
            )
        )
        .astype(str)
    )
    return frame


def build_flip_summary(sentence_frame):
    frame = _confidence_quartiles(sentence_frame)
    rows = []
    group_columns = ["setup", "text_mode", "eeg_control"]
    for keys, values in frame.groupby(group_columns, sort=True):
        rows.append(
            {
                "setup": keys[0],
                "text_mode": keys[1],
                "eeg_control": keys[2],
                "n_seeds": int(values["seed"].nunique()),
                "n_predictions": int(len(values)),
                "prediction_changed_count": int(values["prediction_changed"].sum()),
                "prediction_changed_rate": float(values["prediction_changed"].mean()),
                "favorable_flip_count": int(values["favorable_flip"].sum()),
                "unfavorable_flip_count": int(values["unfavorable_flip"].sum()),
                "net_correct_change": int(
                    values["correct_with_eeg"].sum()
                    - values["correct_without_eeg"].sum()
                ),
                "logit_delta_l2_mean": float(values["logit_delta_l2"].mean()),
                "contribution_to_text_norm_ratio_mean": float(
                    values["contribution_to_text_norm_ratio"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_contribution_comparisons(
    sentence_frame, bootstrap_samples, seed=20260724
):
    """Compare each trained gated model with its own EEG contribution removed."""
    frame = _confidence_quartiles(sentence_frame)
    rows = []
    rng = np.random.default_rng(seed)
    for setup, values in frame.groupby("setup", sort=True):
        values = values.copy()
        values["delta_correct"] = (
            values["correct_with_eeg"] - values["correct_without_eeg"]
        )
        values["prediction_with_eeg_aligned"] = values["prediction_with_eeg"]
        values["prediction_with_eeg_control"] = values["prediction_without_eeg"]
        values["correct_with_eeg_aligned"] = values["correct_with_eeg"]
        values["correct_with_eeg_control"] = values["correct_without_eeg"]
        values["prediction_agreement"] = 1 - values["prediction_changed"]
        for subset_type, subset_name, mask, _ in _subsets(values):
            subset = values.loc[mask]
            if subset.empty:
                continue
            low, high = _cluster_accuracy_interval(
                subset, bootstrap_samples, rng
            )
            per_seed = subset.groupby("seed")["delta_correct"].mean()
            rows.append(
                {
                    "setup": setup,
                    "text_mode": subset["text_mode"].iloc[0],
                    "eeg_control": subset["eeg_control"].iloc[0],
                    "subset_type": subset_type,
                    "subset_name": subset_name,
                    "n_sentences": int(subset["sentence_id"].nunique()),
                    "n_seed_predictions": int(len(subset)),
                    "with_modality_accuracy": float(
                        subset["correct_with_eeg"].mean()
                    ),
                    "without_modality_accuracy": float(
                        subset["correct_without_eeg"].mean()
                    ),
                    "delta_accuracy": float(subset["delta_correct"].mean()),
                    "delta_accuracy_ci95_low": low,
                    "delta_accuracy_ci95_high": high,
                    "prediction_changed_rate": float(
                        subset["prediction_changed"].mean()
                    ),
                    "favorable_flip_count": int(subset["favorable_flip"].sum()),
                    "unfavorable_flip_count": int(
                        subset["unfavorable_flip"].sum()
                    ),
                    "n_seeds_modality_better": int((per_seed > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_paired_frame(sentence_frame):
    aligned = sentence_frame[sentence_frame["eeg_control"] == "aligned"].copy()
    rows = []
    for text_mode in TEXT_MODES:
        aligned_mode = aligned[aligned["text_mode"] == text_mode]
        for control in CONTROLS:
            control_mode = sentence_frame[
                (sentence_frame["text_mode"] == text_mode)
                & (sentence_frame["eeg_control"] == control)
            ]
            keep = [
                "seed",
                "sentence_id",
                "target",
                "prediction_with_eeg",
                "correct_with_eeg",
            ]
            pair = aligned_mode.merge(
                control_mode[keep],
                on=["seed", "sentence_id", "target"],
                suffixes=("_aligned", "_control"),
                validate="one_to_one",
            )
            if len(pair) != len(aligned_mode):
                raise ValueError(
                    f"aligned/{control} rows are incomplete for {text_mode}"
                )
            pair["control"] = control
            pair["delta_correct"] = (
                pair["correct_with_eeg_aligned"]
                - pair["correct_with_eeg_control"]
            )
            pair["prediction_agreement"] = (
                pair["prediction_with_eeg_aligned"]
                == pair["prediction_with_eeg_control"]
            ).astype(int)
            rows.append(pair)
    paired = pd.concat(rows, ignore_index=True)
    paired["confidence_quartile"] = (
        paired.groupby(["text_mode", "control", "seed"])[
            "text_path_confidence"
        ]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=["q1_lowest", "q2", "q3", "q4_highest"],
            )
        )
        .astype(str)
    )
    return paired


def _cluster_accuracy_interval(values, samples, rng):
    """Bootstrap sentences while retaining all seed predictions per sentence."""
    clusters = values.groupby("sentence_id")["delta_correct"].agg(["sum", "count"])
    if not len(clusters):
        return np.nan, np.nan
    sampled = rng.integers(0, len(clusters), size=(int(samples), len(clusters)))
    sums = clusters["sum"].to_numpy()[sampled].sum(axis=1)
    counts = clusters["count"].to_numpy()[sampled].sum(axis=1)
    draws = sums / counts
    return tuple(float(value) for value in np.percentile(draws, [2.5, 97.5]))


def _macro_f1(targets, predictions):
    return float(
        f1_score(
            targets,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        )
    )


def _cluster_macro_f1_interval(values, samples, rng):
    sentence_ids = values["sentence_id"].unique()
    grouped = {
        sentence_id: values[values["sentence_id"] == sentence_id]
        for sentence_id in sentence_ids
    }
    differences = []
    for _ in range(int(samples)):
        sampled = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        target_parts = []
        aligned_parts = []
        control_parts = []
        for sentence_id in sampled:
            rows = grouped[sentence_id]
            target_parts.append(rows["target"].to_numpy())
            aligned_parts.append(rows["prediction_with_eeg_aligned"].to_numpy())
            control_parts.append(rows["prediction_with_eeg_control"].to_numpy())
        targets = np.concatenate(target_parts)
        aligned = np.concatenate(aligned_parts)
        control = np.concatenate(control_parts)
        differences.append(
            _macro_f1(targets, aligned) - _macro_f1(targets, control)
        )
    return tuple(float(value) for value in np.percentile(differences, [2.5, 97.5]))


def _subsets(pair):
    q25 = pair.groupby("seed")["text_path_confidence"].transform(
        lambda values: values.quantile(0.25)
    )
    q50 = pair.groupby("seed")["text_path_confidence"].transform(
        lambda values: values.quantile(0.50)
    )
    subsets = [
        ("all", "all", np.ones(len(pair), dtype=bool), True),
        (
            "text_difficulty",
            "low_confidence_25",
            pair["text_path_confidence"] <= q25,
            True,
        ),
        (
            "text_difficulty",
            "low_confidence_50",
            pair["text_path_confidence"] <= q50,
            True,
        ),
        (
            "text_difficulty",
            "no_eeg_incorrect",
            pair["correct_without_eeg"] == 0,
            True,
        ),
    ]
    for class_name in CLASS_NAMES:
        subsets.append(
            ("true_class", class_name, pair["class_name"] == class_name, False)
        )
    for group in ("short", "medium", "long"):
        subsets.append(
            ("sentence_length", group, pair["length_group"] == group, False)
        )
    for quartile in ("q1_lowest", "q2", "q3", "q4_highest"):
        subsets.append(
            (
                "confidence_quartile",
                quartile,
                pair["confidence_quartile"] == quartile,
                False,
            )
        )
    for count in sorted(pair["reader_count"].unique()):
        subsets.append(
            (
                "reader_count",
                str(int(count)),
                pair["reader_count"] == count,
                False,
            )
        )
    return subsets


def build_subset_comparisons(paired, bootstrap_samples, seed=20260723):
    rows = []
    rng = np.random.default_rng(seed)
    for (text_mode, control), pair in paired.groupby(
        ["text_mode", "control"], sort=True
    ):
        for subset_type, subset_name, mask, include_macro_f1 in _subsets(pair):
            values = pair.loc[mask].copy()
            if values.empty:
                continue
            accuracy_low, accuracy_high = _cluster_accuracy_interval(
                values, bootstrap_samples, rng
            )
            per_seed = values.groupby("seed")["delta_correct"].mean()
            aligned_macro_f1 = _macro_f1(
                values["target"], values["prediction_with_eeg_aligned"]
            )
            control_macro_f1 = _macro_f1(
                values["target"], values["prediction_with_eeg_control"]
            )
            macro_low, macro_high = (np.nan, np.nan)
            if include_macro_f1:
                macro_low, macro_high = _cluster_macro_f1_interval(
                    values, bootstrap_samples, rng
                )
            rows.append(
                {
                    "text_mode": text_mode,
                    "control": control,
                    "subset_type": subset_type,
                    "subset_name": subset_name,
                    "n_sentences": int(values["sentence_id"].nunique()),
                    "n_seed_predictions": int(len(values)),
                    "aligned_accuracy": float(
                        values["correct_with_eeg_aligned"].mean()
                    ),
                    "control_accuracy": float(
                        values["correct_with_eeg_control"].mean()
                    ),
                    "delta_accuracy": float(values["delta_correct"].mean()),
                    "delta_accuracy_ci95_low": accuracy_low,
                    "delta_accuracy_ci95_high": accuracy_high,
                    "aligned_macro_f1": aligned_macro_f1,
                    "control_macro_f1": control_macro_f1,
                    "delta_macro_f1": aligned_macro_f1 - control_macro_f1,
                    "delta_macro_f1_ci95_low": macro_low,
                    "delta_macro_f1_ci95_high": macro_high,
                    "prediction_agreement": float(
                        values["prediction_agreement"].mean()
                    ),
                    "seed_delta_accuracy_mean": float(per_seed.mean()),
                    "seed_delta_accuracy_min": float(per_seed.min()),
                    "seed_delta_accuracy_max": float(per_seed.max()),
                    "n_seeds_aligned_better": int((per_seed > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_stop_decision(comparisons, minimum_delta):
    candidates = []
    priority = comparisons[comparisons["subset_name"].isin(PRIORITY_SUBSETS)]
    for (text_mode, subset_name), values in priority.groupby(
        ["text_mode", "subset_name"], sort=True
    ):
        controls_present = set(values["control"])
        passed = (
            controls_present == set(CONTROLS)
            and bool((values["delta_accuracy"] >= minimum_delta).all())
            and bool((values["delta_accuracy_ci95_low"] > 0).all())
            and bool((values["n_seeds_aligned_better"] >= 2).all())
        )
        candidates.append(
            {
                "text_mode": text_mode,
                "subset_name": subset_name,
                "passed_all_controls": bool(passed),
            }
        )
    detected = any(item["passed_all_controls"] for item in candidates)
    return {
        "alignment_specific_priority_subset_detected": bool(detected),
        "minimum_delta_accuracy": float(minimum_delta),
        "required_controls": list(CONTROLS),
        "required_ci_condition": "delta_accuracy_ci95_low > 0 for every control",
        "required_seed_condition": "aligned better in at least 2 of 3 seeds",
        "priority_subsets": list(PRIORITY_SUBSETS),
        "candidate_results": candidates,
        "interpretation": (
            "A priority text-hard subset passed the exploratory alignment-specific "
            "screen. Confirm it in a new analysis before changing the main conclusion."
            if detected
            else "No priority text-hard subset passed the alignment-specific screen; "
            "close the current pooled classical-feature fusion pipeline."
        ),
    }


def _write_markdown_table(frame, path, title, description):
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(4)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_findings(comparisons, contributions, flips, decision, path):
    full = comparisons[comparisons["subset_name"] == "all"]
    priority = comparisons[
        comparisons["subset_name"].isin(PRIORITY_SUBSETS)
    ]
    aligned_flips = flips[flips["eeg_control"] == "aligned"]
    aligned_contributions = contributions[
        (contributions["eeg_control"] == "aligned")
        & (contributions["subset_name"].isin(PRIORITY_SUBSETS))
    ]
    lines = [
        "# Step 1: saved-prediction closeout analysis",
        "",
        "This analysis uses only previously saved out-of-fold predictions. It does "
        "not retrain or load LaBSE. Text-path confidence is calculated from the "
        "`logits_without_eeg` output of each matched gated model; it is not a saved "
        "confidence score from the separately trained text-only model.",
        "",
        "## Whole-dataset aligned-versus-control results",
        "",
    ]
    for row in full.itertuples(index=False):
        lines.append(
            f"- {row.text_mode}, aligned minus {row.control}: macro-F1 "
            f"{row.delta_macro_f1:+.4f} "
            f"(sentence-cluster bootstrap 95% interval "
            f"[{row.delta_macro_f1_ci95_low:+.4f}, "
            f"{row.delta_macro_f1_ci95_high:+.4f}]); prediction agreement "
            f"{row.prediction_agreement:.2%}."
        )
    lines.extend(["", "## Text-hard subsets", ""])
    for row in priority.itertuples(index=False):
        lines.append(
            f"- {row.text_mode}, {row.subset_name}, aligned minus {row.control}: "
            f"accuracy {row.delta_accuracy:+.4f} "
            f"[{row.delta_accuracy_ci95_low:+.4f}, "
            f"{row.delta_accuracy_ci95_high:+.4f}], "
            f"{row.n_sentences} unique sentences."
        )
    lines.extend(["", "## Aligned EEG with-versus-without contribution", ""])
    for row in aligned_contributions.itertuples(index=False):
        lines.append(
            f"- {row.text_mode}, {row.subset_name}: with-minus-without EEG "
            f"accuracy {row.delta_accuracy:+.4f} "
            f"[{row.delta_accuracy_ci95_low:+.4f}, "
            f"{row.delta_accuracy_ci95_high:+.4f}]; "
            f"{row.favorable_flip_count} favorable and "
            f"{row.unfavorable_flip_count} unfavorable flips."
        )
    lines.extend(["", "## EEG-removal flips over all sentences", ""])
    for row in aligned_flips.itertuples(index=False):
        lines.append(
            f"- {row.text_mode}: {row.prediction_changed_count} of "
            f"{row.n_predictions} seed/sentence predictions changed; "
            f"{row.favorable_flip_count} favorable and "
            f"{row.unfavorable_flip_count} unfavorable."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision["interpretation"],
            "",
            "Class, length, reader-count, confidence-quartile, and per-sentence "
            "tables are exploratory diagnostics. They must not be used to redefine "
            "the primary task after looking at the results.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _plot_subset_deltas(comparisons, path):
    selected = comparisons[
        comparisons["subset_name"].isin(("all",) + PRIORITY_SUBSETS)
    ]
    subset_order = ["all", *PRIORITY_SUBSETS]
    colors = {"shuffled": "#4C78A8", "noise": "#F58518", "zero": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, text_mode in zip(axes, TEXT_MODES):
        mode = selected[selected["text_mode"] == text_mode]
        x = np.arange(len(subset_order))
        width = 0.24
        for offset, control in enumerate(CONTROLS):
            values = mode[mode["control"] == control].set_index("subset_name")
            values = values.reindex(subset_order)
            y = values["delta_accuracy"].to_numpy()
            low = np.maximum(
                0, y - values["delta_accuracy_ci95_low"].to_numpy()
            )
            high = np.maximum(
                0, values["delta_accuracy_ci95_high"].to_numpy() - y
            )
            ax.bar(
                x + (offset - 1) * width,
                y,
                width,
                yerr=np.vstack([low, high]),
                label=control,
                color=colors[control],
                capsize=3,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, [name.replace("_", "\n") for name in subset_order])
        ax.set_title(text_mode)
        ax.set_ylabel("aligned minus control accuracy")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(title="control")
    fig.suptitle("Alignment-specific effects in predefined text-hard subsets")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_flips(flips, path):
    ordered = flips.sort_values(["text_mode", "eeg_control"])
    labels = [
        f"{row.text_mode}\n{row.eeg_control}" for row in ordered.itertuples(index=False)
    ]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x,
        ordered["favorable_flip_count"],
        color="#54A24B",
        label="favorable",
    )
    ax.bar(
        x,
        -ordered["unfavorable_flip_count"],
        color="#E45756",
        label="unfavorable",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("prediction flips across seeds")
    ax.set_title("Changes caused by adding each gated modality")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_closeout_analysis(
    source_run_dir,
    labels_csv,
    features_dir,
    output_dir,
    bootstrap_samples=2000,
    minimum_delta=0.015,
):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "analysis_manifest.json")
    manifest = {
        "analysis_version": "v3_step1_closeout",
        "source_run_dir": os.path.abspath(source_run_dir),
        "labels_csv": os.path.abspath(labels_csv),
        "features_dir": os.path.abspath(features_dir),
        "bootstrap_samples": int(bootstrap_samples),
        "minimum_delta": float(minimum_delta),
        "confidence_definition": (
            "maximum softmax probability from logits_without_eeg in the "
            "matched gated model"
        ),
        "uncertainty_unit": (
            "sentence-cluster bootstrap retaining all seed predictions for each "
            "resampled sentence"
        ),
    }
    if os.path.exists(manifest_path):
        existing = load_json(manifest_path)
        if existing != manifest:
            raise ValueError(
                "the analysis output folder contains a different manifest; "
                "choose a new output tag"
            )
    save_json(manifest, manifest_path)

    indexed = _result_index(source_run_dir)
    _required_results(indexed)
    metadata = _sentence_metadata(labels_csv, features_dir)
    sentence_frame = build_sentence_diagnostics(indexed, metadata)
    flips = build_flip_summary(sentence_frame)
    contributions = build_contribution_comparisons(
        sentence_frame, bootstrap_samples=bootstrap_samples
    )
    paired = build_paired_frame(sentence_frame)
    comparisons = build_subset_comparisons(
        paired, bootstrap_samples=bootstrap_samples
    )
    decision = evaluate_stop_decision(comparisons, minimum_delta)

    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    sentence_frame.to_csv(
        os.path.join(tables_dir, "sentence_diagnostics.csv"), index=False
    )
    paired.to_csv(os.path.join(tables_dir, "paired_predictions.csv"), index=False)
    flips.to_csv(os.path.join(tables_dir, "prediction_flips.csv"), index=False)
    contributions.to_csv(
        os.path.join(tables_dir, "modality_contribution_subsets.csv"),
        index=False,
    )
    comparisons.to_csv(
        os.path.join(tables_dir, "subset_control_comparisons.csv"), index=False
    )
    save_json(
        flips.to_dict(orient="records"),
        os.path.join(tables_dir, "prediction_flips.json"),
    )
    save_json(
        comparisons.to_dict(orient="records"),
        os.path.join(tables_dir, "subset_control_comparisons.json"),
    )
    save_json(
        contributions.to_dict(orient="records"),
        os.path.join(tables_dir, "modality_contribution_subsets.json"),
    )
    save_json(decision, os.path.join(tables_dir, "decision.json"))
    _write_markdown_table(
        flips,
        os.path.join(tables_dir, "prediction_flips.md"),
        "EEG-removal prediction flips",
        "Favorable and unfavorable changes are counted over all held-out "
        "seed/sentence predictions.",
    )
    _write_markdown_table(
        contributions,
        os.path.join(tables_dir, "modality_contribution_subsets.md"),
        "Each gated model with and without its modality contribution",
        "These within-model comparisons show whether adding a modality changed "
        "accuracy. They do not by themselves establish alignment-specific value.",
    )
    _write_markdown_table(
        comparisons,
        os.path.join(tables_dir, "subset_control_comparisons.md"),
        "Aligned EEG against controls by subset",
        "Accuracy intervals resample unique sentences while retaining all seed "
        "predictions. Macro-F1 intervals are produced for the complete and "
        "predefined text-hard subsets.",
    )
    _write_findings(
        comparisons,
        contributions,
        flips,
        decision,
        os.path.join(tables_dir, "findings.md"),
    )
    _plot_subset_deltas(
        comparisons, os.path.join(plots_dir, "text_hard_subset_deltas.png")
    )
    _plot_flips(flips, os.path.join(plots_dir, "prediction_flips.png"))
    return comparisons, decision
