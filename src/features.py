"""Classical sentence-level EEG features."""

import warnings

import numpy as np
from scipy.stats import kurtosis, skew


BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
RAW_STATS = [
    "mean",
    "std",
    "var",
    "min",
    "max",
    "ptp",
    "median",
    "iqr",
    "skew",
    "kurtosis",
    "rms",
    "mav",
    "line_length",
    "zcr",
    "hjorth_mobility",
    "hjorth_complexity",
]
N_FAMILIES = len(RAW_STATS) + len(BANDS)


def _safe(func, *args, **kwargs):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _zcr(signal):
    centered = signal - _safe(np.nanmean, signal, axis=1, keepdims=True)
    filled = np.where(np.isnan(centered), 0.0, centered)
    crossings = np.sum(np.abs(np.diff(np.sign(filled), axis=1)) > 0, axis=1)
    return crossings / max(signal.shape[1] - 1, 1)


def _hjorth(signal):
    first = np.diff(signal, axis=1)
    second = np.diff(first, axis=1)
    v0 = _safe(np.nanvar, signal, axis=1)
    v1 = _safe(np.nanvar, first, axis=1)
    v2 = _safe(np.nanvar, second, axis=1)
    with np.errstate(all="ignore"):
        mobility = np.sqrt(v1 / v0)
        complexity = np.sqrt(v2 / v1) / mobility
    return mobility, complexity


def _line_length(signal, mode):
    differences = np.abs(np.diff(signal, axis=1))
    if mode == "sum":
        return _safe(np.nansum, differences, axis=1)
    if mode != "normalized":
        raise ValueError("line_length must be 'normalized' or 'sum'")
    valid = np.isfinite(differences).sum(axis=1)
    total = _safe(np.nansum, differences, axis=1)
    return np.divide(total, valid, out=np.full_like(total, np.nan), where=valid > 0)


def raw_feature_block(signal, line_length="normalized"):
    """Return the 16 raw statistics as ``[families, channels]``."""
    mobility, complexity = _hjorth(signal)
    minimum = _safe(np.nanmin, signal, axis=1)
    maximum = _safe(np.nanmax, signal, axis=1)
    table = {
        "mean": _safe(np.nanmean, signal, axis=1),
        "std": _safe(np.nanstd, signal, axis=1),
        "var": _safe(np.nanvar, signal, axis=1),
        "min": minimum,
        "max": maximum,
        "ptp": maximum - minimum,
        "median": _safe(np.nanmedian, signal, axis=1),
        "iqr": (
            _safe(np.nanpercentile, signal, 75, axis=1)
            - _safe(np.nanpercentile, signal, 25, axis=1)
        ),
        "skew": _safe(skew, signal, axis=1, nan_policy="omit"),
        "kurtosis": _safe(kurtosis, signal, axis=1, nan_policy="omit"),
        "rms": np.sqrt(_safe(np.nanmean, signal ** 2, axis=1)),
        "mav": _safe(np.nanmean, np.abs(signal), axis=1),
        "line_length": _line_length(signal, line_length),
        "zcr": _zcr(signal),
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }
    return np.stack([table[name] for name in RAW_STATS])


def feature_names(n_channels):
    names = []
    for family in RAW_STATS:
        names.extend(f"raw_{family}_ch{channel}" for channel in range(n_channels))
    for band in BANDS:
        names.extend(f"bandmean_{band}_ch{channel}" for channel in range(n_channels))
    return names


def sentence_features(sentence, n_channels, line_length="normalized"):
    """Flatten raw statistics and ZuCo band means in family-major order."""
    raw = sentence.get("raw")
    if raw is None or raw.shape[0] != n_channels:
        raw_block = np.full((len(RAW_STATS), n_channels), np.nan)
    else:
        raw_block = raw_feature_block(raw, line_length=line_length)

    band_block = []
    for band in BANDS:
        values = sentence.get("bands", {}).get(band)
        if values is None or len(values) != n_channels:
            values = np.full(n_channels, np.nan)
        band_block.append(np.asarray(values, dtype=np.float64))
    return np.concatenate([raw_block, np.stack(band_block)], axis=0).reshape(-1).astype(np.float32)


def infer_channels(sentences):
    for sentence in sentences:
        if sentence.get("raw") is not None:
            return sentence["raw"].shape[0]
    return None
