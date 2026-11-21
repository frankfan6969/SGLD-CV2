"""Explicit, cached UCI acquisition and deterministic model preprocessing."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from scipy.special import expit
from .models import LogisticRegression, GaussianRegression

URLS = {
    "german": "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
    "year": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(dataset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{dataset}.zip"
    if not destination.exists():
        partial = destination.with_suffix(".part")
        with urllib.request.urlopen(URLS[dataset], timeout=120) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("downloaded archive failed CRC check")
        partial.replace(destination)
    provenance = {"url": URLS[dataset], "sha256": file_hash(destination), "bytes": destination.stat().st_size}
    provenance_path = directory / f"{dataset}.source.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text())["sha256"] != provenance["sha256"]:
        raise ValueError("cached source hash changed")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return destination


def german_credit(path, *, prior_precision=1., intercept=False):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.split("/")[-1] == "german.data")
        rows = [line.split() for line in archive.read(member).decode().splitlines() if line.strip()]
    raw = np.array(rows)
    if raw.shape != (1000, 21) or not np.isin(raw[:, -1], ["1", "2"]).all():
        raise ValueError("unexpected German Credit shape/labels")
    columns, names, transforms = [], [], {}
    for j in range(20):
        values = raw[:, j]
        if all(not str(v).startswith("A") for v in values):
            numeric = values.astype(float)
            mean, scale = float(numeric.mean()), float(numeric.std()) or 1.
            columns.append((numeric - mean) / scale)
            names.append(f"attribute_{j+1}")
            transforms[str(j + 1)] = {"mean": mean, "scale": scale}
        else:
            levels = sorted(set(values))
            for level in levels:
                columns.append((values == level).astype(float))
                names.append(f"attribute_{j+1}_{level}")
            transforms[str(j + 1)] = {"levels": levels, "drop_reference": False}
    if intercept:
        columns.append(np.ones(1000)); names.append("intercept")
    model = LogisticRegression(np.column_stack(columns), (raw[:, -1] == "1").astype(int), prior_precision)
    return model, {"dataset": "UCI German Credit", "source_sha256": file_hash(path),
                   "feature_names": names, "preprocessing": transforms, "intercept": intercept,
                   "labels": "1 good -> 1; 2 bad -> 0", "N": model.N, "d": model.d}


def synthetic_logistic(*, N=1000, d=20, design_kappa=100, seed=221, prior_precision=1.):
    if N <= d or d < 2 or design_kappa < 1:
        raise ValueError("invalid synthetic design")
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    spectrum = np.geomspace(1, 1 / design_kappa, d)
    X = rng.normal(size=(N, d)) @ (rotation * np.sqrt(spectrum)).T
    theta = rng.normal(size=d)
    theta /= np.std(X @ theta)
    y = rng.binomial(1, expit(X @ theta))
    return LogisticRegression(X, y, prior_precision), {
        "dataset": "synthetic logistic", "N": N, "d": d, "seed": seed,
        "requested_design_covariance_kappa": design_kappa, "realized_design_gram_kappa": float(np.linalg.cond(X.T @ X)),
        "true_theta": theta.tolist(), "prior_precision": prior_precision}


def prepare_year(archive_path, destination, *, rows=463715, standardize=False, intercept=False):
    """Read ONLY the official training prefix; optional smaller validation subset.

    Nested UCI zip layouts are supported without extracting untrusted paths.
    float64 memmap output avoids holding text plus multiple dense arrays.
    """
    if not 2 <= rows <= 463715:
        raise ValueError("rows must lie within the official 463715-row training prefix")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    d = 90 + int(intercept)
    X = np.lib.format.open_memmap(destination / "X.npy", mode="w+", dtype="float64", shape=(rows, d))
    y = np.lib.format.open_memmap(destination / "y.npy", mode="w+", dtype="float64", shape=(rows,))
    with zipfile.ZipFile(archive_path) as outer:
        txt = next((n for n in outer.namelist() if n.endswith(".txt")), None)
        if txt is None:
            nested_name = next(n for n in outer.namelist() if n.endswith(".zip"))
            nested_path = destination / "source_inner.zip"
            with outer.open(nested_name) as source, nested_path.open("wb") as out:
                while block := source.read(1024 * 1024): out.write(block)
            inner = zipfile.ZipFile(nested_path)
            txt = next(n for n in inner.namelist() if n.endswith(".txt"))
            archive = inner
        else:
            inner, archive = None, outer
        try:
            with archive.open(txt) as source:
                for i in range(rows):
                    values = np.fromstring(source.readline().decode(), sep=",")
                    if len(values) != 91 or not np.isfinite(values).all():
                        raise ValueError(f"invalid Year row {i}")
                    y[i], X[i, :90] = values[0], values[1:]
                    if intercept: X[i, -1] = 1.
        finally:
            if inner is not None: inner.close()
    raw_gram = np.zeros((90, 90)); total = np.zeros(90)
    for i in range(0, rows, 8192):
        block = X[i:i+8192, :90]
        raw_gram += block.T @ block
        total += block.sum(axis=0)
    mean = total / rows
    scales = np.sqrt(np.maximum(np.diag(raw_gram) / rows - mean**2, 0))
    scales[scales == 0] = 1.
    target_mean = float(y.mean()) if standardize else 0.
    if standardize:
        for i in range(0, rows, 8192): X[i:i+8192, :90] = (X[i:i+8192, :90] - mean) / scales
        y[:] -= target_mean
    X.flush(); y.flush()
    meta = {"dataset": "UCI YearPredictionMSD", "rows": rows, "d": d,
            "official_training_rows": 463715, "validation_subset": rows != 463715,
            "source_sha256": file_hash(archive_path), "standardize": standardize, "intercept": intercept,
            "feature_means": mean.tolist(), "feature_scales": scales.tolist(), "target_center": target_mean,
            "raw_gram_condition_number": float(np.linalg.cond(raw_gram)),
            "model_note": "isotropic ridge prior in configured coordinates; scaled and raw models differ"}
    (destination / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def year_model(directory, *, prior_precision=1., noise_variance=1.):
    directory = Path(directory)
    # Immutable no-copy arrays allow the full 334 MB design to remain memory mapped.
    model = GaussianRegression.__new__(GaussianRegression)
    model.X = np.load(directory / "X.npy", mmap_mode="r")
    model.y = np.load(directory / "y.npy", mmap_mode="r")
    if model.X.ndim != 2 or model.y.shape != (len(model.X),) or min(model.X.shape) < 1:
        raise ValueError("invalid prepared Year array shapes")
    model.N, model.d = model.X.shape
    if not np.isfinite([prior_precision,noise_variance]).all() or prior_precision <= 0 or noise_variance <= 0:
        raise ValueError("positive prior precision and fixed noise variance required")
    for first in range(0,model.N,8192):
        if not np.isfinite(model.X[first:first+8192]).all() or not np.isfinite(model.y[first:first+8192]).all():
            raise ValueError("nonfinite prepared Year data")
    model.prior_precision, model.noise_variance = float(prior_precision), float(noise_variance)
    return model, json.loads((directory / "metadata.json").read_text())
