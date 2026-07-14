"""Extract and cache ZuCo sentence-level classical EEG features."""

import argparse
import glob
import json
import os

import numpy as np

from src.features import feature_names, infer_channels, sentence_features
from src.labels import label_lookup, match_sentence
from src.utils import save_json
from src.zuco_io import iter_sentences, subject_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Cache classical features from ZuCo .mat files.")
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--line-length",
        choices=["normalized", "sum"],
        default="normalized",
        help="normalized removes the sentence-duration dependence of cumulative line length",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_subject(path, lookup, line_length):
    sentences = list(iter_sentences(path))
    n_channels = infer_channels(sentences)
    if n_channels is None:
        return None
    rows, sentence_ids, labels, contents = [], [], [], []
    for sentence in sentences:
        sentence_id, label = match_sentence(sentence["content"], lookup)
        if sentence_id is None:
            continue
        values = sentence_features(sentence, n_channels, line_length=line_length)
        if not np.isfinite(values).any():
            continue
        rows.append(values)
        sentence_ids.append(sentence_id)
        labels.append(label)
        contents.append(sentence["content"])
    if not rows:
        return None
    return {
        "X": np.vstack(rows).astype(np.float32),
        "sentence_id": np.asarray(sentence_ids, dtype=np.int32),
        "label": np.asarray(labels, dtype=np.int8),
        "content": np.asarray(contents, dtype=object),
        "n_channels": n_channels,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "feature_manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        with open(manifest_path) as handle:
            existing_manifest = json.load(handle)
        if existing_manifest.get("line_length") != args.line_length:
            raise SystemExit(
                "this feature folder uses a different line-length definition; "
                "choose a new --out-dir or pass --overwrite"
            )
    lookup = label_lookup(args.labels_csv)
    files = sorted(glob.glob(os.path.join(args.mat_dir, "*SR*.mat")))
    if not files:
        raise SystemExit(f"no *SR*.mat files found in {args.mat_dir}")
    print(f"{len(files)} subject files, {len(lookup)} labelled sentences")

    counts = {}
    expected_names = None
    for path in files:
        subject = subject_from_path(path)
        out_path = os.path.join(args.out_dir, f"{subject}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            cached = np.load(out_path, allow_pickle=True)
            counts[subject] = int(len(cached["label"]))
            print(f"  skip {subject}: {counts[subject]} cached sentences")
            continue

        result = extract_subject(path, lookup, args.line_length)
        if result is None:
            print(f"  {subject}: no usable EEG")
            continue
        names = feature_names(result.pop("n_channels"))
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f"channel layout differs for {subject}")
        temporary = out_path + ".tmp.npz"
        np.savez_compressed(temporary, **result)
        os.replace(temporary, out_path)
        counts[subject] = int(len(result["label"]))
        print(f"  {subject}: {result['X'].shape[0]} sentences x {result['X'].shape[1]} features")

    names_path = os.path.join(args.out_dir, "feature_names.json")
    if expected_names is not None:
        save_json(expected_names, names_path)
    elif not os.path.exists(names_path):
        raise RuntimeError("feature_names.json is missing and every subject was skipped")
    with open(names_path) as handle:
        names = json.load(handle)
    save_json(
        {
            "line_length": args.line_length,
            "n_subjects": len(counts),
            "n_features": len(names),
            "subject_sentence_counts": counts,
            "labels_csv": os.path.basename(args.labels_csv),
        },
        manifest_path,
    )
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
