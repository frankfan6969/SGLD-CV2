from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_german_p1_stage import schedule,validate_execution


def inputs():
    return (json.loads((ROOT/"configs/p1_german_amendment_v1.json").read_text()),
            json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text()))


def test_amendment_schedule_has_only_reserved_seed_role():
    e,c=inputs();validate_execution(e,c);jobs=schedule(e)
    assert len(jobs)==8
    assert {j["seed"] for j in jobs}==set(c["seeds"]["pilot_amendment_reserved"])
    assert len({(j["candidate"]["id"],j["seed"]) for j in jobs})==8
    assert schedule(e)==jobs


def test_heldout_and_cross_role_seed_execution_rejected():
    e,c=inputs();e["stage"]="heldout"
    with pytest.raises(ValueError):validate_execution(e,c)
    e,c=inputs();e["seeds"]=c["seeds"]["selection"]
    with pytest.raises(ValueError):validate_execution(e,c)


def test_selection_cannot_drop_matched_cells_or_raise_budget():
    e,c=inputs();e.update(stage="selection",seeds=c["seeds"]["selection"],budget=deepcopy(c["budgets"]["selection"]))
    e["candidates"]=[p for p in c["pilot_candidates"] if p["arm"].startswith("full_") and p["coefficient"] in [.1,.4]]
    assert validate_execution(e,c)
    bad=deepcopy(e);bad["candidates"].pop()
    with pytest.raises(ValueError):validate_execution(bad,c)
    bad=deepcopy(e);bad["budget"]["replicas"]=64
    with pytest.raises(ValueError):validate_execution(bad,c)
