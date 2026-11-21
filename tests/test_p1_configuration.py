from copy import deepcopy
from itertools import combinations
import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("german_freeze", ROOT/"scripts/freeze_german_p1.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


@pytest.mark.parametrize("b", [1, 2, 4, 5])
def test_exact_li_planning_second_moment(b):
    g = np.array([[1.,-2.], [3.,4.], [-1.,2.], [.5,-.3], [2.,1.]])
    enumerated = np.mean([g[list(indices)].mean(axis=0)**2 for indices in combinations(range(5), b)], axis=0)
    assert_allclose(freeze.raw_second_moment(g,b), enumerated, atol=1e-14)


def test_curvature_scale_and_public_h_units():
    H = np.array([[5.,1.], [1.,2.]])
    assert_allclose(freeze.curvature_scale(H,np.linalg.inv(H)), 1., atol=1e-14)
    assert_allclose(freeze.curvature_scale(H,np.eye(2)), np.linalg.eigvalsh(H)[-1])
    assert_allclose(freeze.curvature_scale(H,3*np.eye(2)), 3*np.linalg.eigvalsh(H)[-1])
    with pytest.raises(ValueError,match="SPD"):
        freeze.curvature_scale(H,np.diag([1.,-1.]))


def test_frozen_config_candidates_and_caps():
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    with np.load(ROOT/"results/p1_german_configuration_v2/geometry_and_li_planning.npz") as data:
        geometry={k:data[k] for k in data.files}
    assert freeze.validate_config(c,geometry)
    assert len(c["pilot_candidates"]) == 45
    for candidate in c["pilot_candidates"]:
        assert_allclose(candidate["legacy_step_size"],2*candidate["h"])
        if candidate["method"]=="ula":
            assert candidate["batch_size"] is None
    assert not c["pilot_execution_started"]
    assert c["budgets"]["selection"]["replicas"] == c["budgets"]["heldout"]["replicas"]
    for phase in ["selection","heldout"]:
        stage=c["budgets"][phase]
        maximum=(3*stage["max_primary_configs_per_arm"]+3*stage["max_li_configs_per_arm"]+
                 stage["max_identity_sgld_configs"])*len(c["seeds"][phase])
        assert stage["maximum_candidate_seed_runs"] == maximum


@pytest.mark.parametrize("change", ["seed", "step", "candidate", "initialization", "replicas"])
def test_freeze_validator_rejects_design_changes(change):
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    with np.load(ROOT/"results/p1_german_configuration_v2/geometry_and_li_planning.npz") as data:
        geometry={k:data[k] for k in data.files}
    if change=="seed":c["seeds"]["heldout"][0]=c["seeds"]["pilot"][0]
    elif change=="step":c["fixed_metrics"]["full"]["h_grid"][0]*=2
    elif change=="candidate":c["pilot_candidates"][0]["h"]*=2
    elif change=="initialization":c["li2016"]["initial_v"]="planning second moment"
    else:c["budgets"]["selection"]["replicas"]=128
    with pytest.raises(ValueError):freeze.validate_config(c,geometry)
