"""Validate the feature cache and print its sentence-level summary."""

import argparse
import json

from src.data import load_multimodal_data


def main():
    parser = argparse.ArgumentParser(description="Inspect aligned ZuCo text and EEG data.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--features-dir", required=True)
    args = parser.parse_args()
    data = load_multimodal_data(args.labels_csv, args.features_dir)
    print(json.dumps(data.summary(), indent=2))


if __name__ == "__main__":
    main()
