"""Method-independent streams: distinct replica IDs remain independent."""
import numpy as np

BATCH = 1
NOISE = 2
INITIALIZATION = 3
REFERENCE = 4
BASELINE = 5
DATASET = 6
TIMING_ORDER = 7


def generator(seed: int, replica_id: int, purpose: int):
    return np.random.default_rng(np.random.SeedSequence([seed, replica_id, purpose]))

