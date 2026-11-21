"""Validate and fingerprint the phase-one protocol without adaptive defaults."""
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_seed_inventory(root):
    """Declared seeds in legacy configs/metadata and saved-result filenames.

    Phase-one validation runs are explicitly excluded so deterministic reruns
    do not make validation seeds look like previously inspected selection seeds.
    """
    root, inventory = Path(root), set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "seed" or key.endswith("seeds"):
                    candidates = item if isinstance(item, list) else [item]
                    inventory.update(s for s in candidates if type(s) is int)
                elif isinstance(item, dict):
                    visit(item)
                elif isinstance(item, list):
                    for child in item:
                        if isinstance(child, dict):
                            visit(child)
    paths = list((root/"configs").rglob("*.json")) + list((root/"results").rglob("*.json"))
    for path in paths:
        if any(part.startswith(("phase1", "p0_")) for part in path.relative_to(root).parts):
            continue
        inventory.update(int(s) for s in re.findall(r"seed(\d+)", path.name))
        if path.parent == root/"configs" or path.name in {"protocol.json", "manifest.json", "frozen_selection.json"}:
            visit(json.loads(path.read_text()))
    return inventory


def validate_protocol(config):
    if config.get("purpose") not in {"phase1_correctness_and_protocol_freeze", "p0_updated_plan_completion"}:
        raise ValueError("wrong phase-one purpose")
    if config.get("dtype") != "float64" or config.get("blas_threads") != 1:
        raise ValueError("phase-one hardware/precision convention changed")
    seen = set()
    for label, seeds in config["seeds"].items():
        if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds) or seen.intersection(seeds)):
            raise ValueError(f"seed role overlap or invalid seeds: {label}")
        seen.update(seeds)
    if seen.intersection(config["reference"]["german"]["seeds"]):
        raise ValueError("new seeds overlap historical reference seeds")
    v = config["validation"]
    if (v["checkpoints"][0] != 0 or v["checkpoints"][-1] != v["steps"] or
            sorted(set(v["checkpoints"])) != v["checkpoints"] or v["replicas"] < 2 or
            not 1 <= v["batch_size"] <= v["N"]):
        raise ValueError("invalid frozen validation grid")
    if config["accuracy"]["replica_rule"] != "one endpoint per independent replica at each checkpoint":
        raise ValueError("finite-K evaluation must use replica endpoints")
    if config["accuracy"]["subtract_reference_baseline"]:
        raise ValueError("reference-reference distance cannot be subtracted")
    if config["functionals"]["burn_in"] != 0:
        raise ValueError("phase-one functional burn-in changed")
    if config["adaptive_baseline"]["initial_v"] != "zeros independently for each replica":
        raise ValueError("adaptive baseline initialization changed")
    required = {"identity_sgld", "identity_cv1", "identity_cv2", "identity_ula",
                "full_sgld", "full_cv1", "full_cv2", "full_ula", "psgld", "pcv1", "pcv2"}
    if config.get("version") == 2:
        required |= {"diagonal_cv1", "diagonal_cv2", "diagonal_ula"}
        if config["fixed_metric_regularization"] != {"relative_ridge": 1e-10, "absolute_ridge": 0.0}:
            raise ValueError("fixed ridge convention differs from frozen P0 v2")
        if config["uncertainty"]["reference_resampling"] != "none; report independent HMC MCSE/disagreement separately":
            raise ValueError("cannot iid-bootstrap correlated HMC reference rows")
    if config.get("sampling_with_replacement") is not False:
        raise ValueError("frozen protocol uses sampling without replacement")
    if set(config["arms"]) != required:
        raise ValueError("baseline/control arms incomplete")
    for name, arm in config["arms"].items():
        expected_method = name.split("_")[-1] if "_" in name else name[1:]
        expected_metric = ("identity" if name.startswith("identity_") else
                           "full_hessian" if name.startswith("full_") else
                           "fixed_diagonal" if name.startswith("diagonal_") else "rmsprop")
        if arm != {"method": expected_method, "metric": expected_metric}:
            raise ValueError(f"arm definition changed: {name}")
    return config


def make_lock(protocol_path, root, assets=()):
    root = Path(root).resolve()
    validate_protocol(json.loads(Path(protocol_path).read_text()))
    paths = [Path(protocol_path), *sorted((root / "src/sgld_cv").glob("*.py")),
             root / "scripts/validate_phase1.py", *map(Path, assets)]
    return {"schema": 1, "files": {str(p.resolve().relative_to(root)): sha256(p) for p in paths}}


def verify_lock(lock, root):
    for name, expected in lock["files"].items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"frozen source/protocol/asset changed: {name}")
