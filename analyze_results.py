"""Analyze saved controlled predictions without loading or training a model."""

import argparse
import os

from src.closeout_analysis import run_closeout_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run the Step 1 closeout analysis on saved OOF predictions."
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--results-base", required=True)
    parser.add_argument("--analysis-tag", default="v3_closeout_analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--minimum-delta", type=float, default=0.015)
    args = parser.parse_args()

    output_dir = os.path.join(args.results_base, args.analysis_tag)
    print(f"source run: {args.source_run_dir}")
    print(f"analysis directory: {output_dir}")
    comparisons, decision = run_closeout_analysis(
        source_run_dir=args.source_run_dir,
        labels_csv=args.labels_csv,
        features_dir=args.features_dir,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
        minimum_delta=args.minimum_delta,
    )
    full = comparisons[comparisons["subset_name"] == "all"]
    print("\nWhole-dataset aligned-versus-control comparisons:")
    print(
        full[
            [
                "text_mode",
                "control",
                "delta_macro_f1",
                "delta_macro_f1_ci95_low",
                "delta_macro_f1_ci95_high",
            ]
        ].to_string(index=False)
    )
    print("\n" + decision["interpretation"])
    print(f"\ncomplete -> {output_dir}")


if __name__ == "__main__":
    main()
