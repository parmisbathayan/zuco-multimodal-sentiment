"""Sentence-label matching shared by feature extraction and training."""

import re

import pandas as pd


def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_labels(path):
    data = pd.read_csv(path)
    required = {"sentence_id", "sentence", "sentiment_label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    data = data[list(required)].dropna().copy()
    data["sentence_id"] = data["sentence_id"].astype(int)
    data["sentiment_label"] = data["sentiment_label"].astype(int)
    if data["sentence_id"].duplicated().any():
        raise ValueError("sentence_id must be unique")
    return data.sort_values("sentence_id").reset_index(drop=True)


def label_lookup(path):
    data = load_labels(path)
    return {
        normalize_text(row.sentence): (int(row.sentence_id), int(row.sentiment_label))
        for row in data.itertuples()
    }


def match_sentence(content, lookup):
    return lookup.get(normalize_text(content), (None, None))
