"""Rebuild summary tables and plots from a completed run folder."""

import argparse

from src.reporting import build_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize a ZuCo multimodal run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    summary = build_summary(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
