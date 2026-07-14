"""Read ZuCo Task 1 sentence structures from MATLAB files."""

import re
import warnings

import numpy as np

from .features import BANDS


def subject_from_path(path):
    match = re.search(r"results([A-Za-z0-9]+)_SR", str(path))
    if match:
        return match.group(1)
    return str(path).split("/")[-1].replace("results", "").replace("_SR.mat", "")


def _orient(array):
    array = np.squeeze(np.asarray(array, dtype=np.float64))
    if array.ndim != 2 or min(array.shape) < 2:
        return None
    if array.shape[0] > array.shape[1]:
        array = array.T
    if array.shape[0] > 256:
        warnings.warn(f"unexpected EEG shape {array.shape}; check orientation")
    return array


def iter_sentences(path):
    try:
        import h5py

        if h5py.is_hdf5(path):
            yield from _iter_hdf5(path)
            return
    except ImportError:
        pass
    yield from _iter_scipy(path)


def _iter_hdf5(path):
    import h5py

    with h5py.File(path, "r") as handle:
        if "sentenceData" not in handle:
            raise KeyError(f"sentenceData is missing from {path}")
        data = handle["sentenceData"]
        content_refs = np.asarray(data["content"]).flatten()
        raw_refs = np.asarray(data["rawData"]).flatten() if "rawData" in data else None
        band_refs = {
            band: np.asarray(data[f"mean_{band}"]).flatten()
            for band in BANDS
            if f"mean_{band}" in data
        }

        for index, content_ref in enumerate(content_refs):
            codes = np.asarray(handle[content_ref]).flatten()
            content = "".join(chr(int(code)) for code in codes if int(code) > 0).strip()
            raw = None
            if raw_refs is not None and index < len(raw_refs) and raw_refs[index]:
                raw = _orient(np.asarray(handle[raw_refs[index]], dtype=np.float64))
            bands = {}
            for band, refs in band_refs.items():
                if index < len(refs) and refs[index]:
                    bands[band] = np.asarray(handle[refs[index]], dtype=np.float64).flatten()
            yield {"content": content, "raw": raw, "bands": bands}


def _iter_scipy(path):
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    for sentence in np.atleast_1d(data["sentenceData"]):
        content = str(getattr(sentence, "content", "") or "").strip()
        raw = getattr(sentence, "rawData", None)
        raw = _orient(raw) if raw is not None and np.size(raw) else None
        bands = {}
        for band in BANDS:
            values = getattr(sentence, f"mean_{band}", None)
            if values is not None and np.size(values):
                bands[band] = np.asarray(values, dtype=np.float64).flatten()
        yield {"content": content, "raw": raw, "bands": bands}
