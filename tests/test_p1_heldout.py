import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from p1_heldout_bootstrap import prepare_projected,weighted_sw2,bootstrap_design
from sgld_cv.transport import AccuracyMetric
from run_german_p1_heldout import validate_execution


def test_weighted_projection_bootstrap_matches_literal_replica_duplicates():
    rng=np.random.default_rng(18);x=rng.normal(size=(16,3));refs=rng.normal(size=(2,32,3))
    directions=rng.normal(size=(3,7));directions/=np.linalg.norm(directions,axis=0)
    metric=AccuracyMetric("standardized_sliced_w2",np.zeros(3),np.eye(3),directions)
    prepared=prepare_projected(x@directions,refs@directions)
    for ids in [np.arange(16),rng.integers(0,16,16),np.zeros(16,dtype=int)]:
        counts=np.bincount(ids,minlength=16)
        actual=weighted_sw2(*prepared,counts)
        expected=[metric.distance(x[ids],ref) for ref in refs]
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_bootstrap_resamples_whole_replicas_and_seed_clusters():
    outer,counts=bootstrap_design(8,512,20,14)
    assert outer.shape==(20,8) and counts.shape==(20,8,512)
    assert (counts.sum(axis=-1)==512).all()
    assert np.any(np.apply_along_axis(lambda a:len(set(a)),1,outer)<8)


def test_heldout_union_cannot_reoptimize_candidates_or_consume_selection_seeds():
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    s=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    q=json.loads((ROOT/"results/p1_german_selection_v1/final_choices.json").read_text())
    ids={p["choice"]["candidate_id"] for p in q["choices"] if p["choice"]}|{p["candidate_id"] for p in q["matched_cells"]}
    e=dict(stage="heldout",seeds=c["seeds"]["heldout"],budget=c["budgets"]["heldout"],
           candidates=[p for p in s["candidates"] if p["id"] in ids])
    assert validate_execution(e,c)
    e["seeds"]=c["seeds"]["selection"]
    with pytest.raises(ValueError):validate_execution(e,c)
    e["seeds"]=c["seeds"]["heldout"];e["candidates"]=e["candidates"][:-1]
    with pytest.raises(ValueError):validate_execution(e,c)
