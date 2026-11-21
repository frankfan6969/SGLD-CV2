import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from p1_bounded_sampler import bounded_replicas
from run_german_p1_pilot import make_schedule
from sgld_cv.anchors import supplied_anchor
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.models import LogisticRegression,GaussianRegression
from sgld_cv.samplers import run_with_theory_step


@pytest.mark.parametrize("method",["sgld","cv1","cv2","ula"])
@pytest.mark.parametrize("geometry",["identity","full","li"])
def test_bounded_runner_matches_frozen_core(method,geometry):
    rng=np.random.default_rng(17)
    model=LogisticRegression(rng.normal(size=(20,3)),rng.integers(0,2,20))
    anchor=supplied_anchor(np.zeros(3),provenance="test")
    estimator=prepare_estimator(model,method,anchor)
    f=FunctionalSpec.from_geometry(anchor.theta,model.full_hessian(anchor.theta))
    matrix=np.array([[1.,.2,0],[.2,2.,.1],[0,.1,1.5]])
    bounded_options={}; core_options={}
    if geometry=="full":
        bounded_options={"matrix":matrix};core_options={"preconditioner":"frozen_full","frozen_matrix":matrix}
    elif geometry=="li":
        bounded_options={"alpha":.99,"epsilon":1e-5};core_options={"preconditioner":"rmsprop","alpha":.99,"epsilon":1e-5}
    common=dict(initial=anchor.theta,h=1e-6,batch_size=4,replicas=4,steps=12,seed=55,checkpoints=[3,7,12],functionals=f)
    bounded=bounded_replicas(estimator,transform=f.transform,max_radius=100*np.sqrt(3),**common,**bounded_options)
    core=run_with_theory_step(estimator,record_adaptive_state=True,**common,**core_options)
    assert not bounded["failures"] and bounded["stop_reason"]=="complete"
    for saved in bounded["saved"]:
        j=list(core.checkpoints).index(saved["iteration"])
        assert_array_equal(saved["endpoints"],core.states[j])
        assert_allclose(saved["functional_averages"],core.functional_averages[j],atol=1e-14)
        assert saved["cost"]["operations"]==core.costs[j]["operations"]
    if geometry=="li":assert_array_equal(bounded["adaptive_final_v"],core.adaptive_final_v)


def test_budget_does_not_score_partial_ensemble_or_invent_checkpoint():
    model=GaussianRegression([[1.],[2.],[3.]],[0,0,0])
    est=prepare_estimator(model,"sgld")
    f=lambda x:np.c_[x,x*x]
    run=bounded_replicas(est,initial=np.zeros(1),h=.001,batch_size=2,replicas=3,steps=10,seed=1,
        checkpoints=[2,10],functionals=f,transform=np.eye(1),max_radius=100,stage_deadline=0.)
    assert run["completed_iterations"]==0
    assert run["stop_reason"]=="stage_budget_censored"
    assert run["saved"]==[]
    assert run["terminal_cost"]["operations"].get("transitions",0)==0


def test_failed_replica_is_never_filtered():
    model=GaussianRegression([[1.],[2.],[3.]],[0,0,0])
    est=prepare_estimator(model,"sgld")
    run=bounded_replicas(est,initial=np.zeros(1),h=1.,batch_size=2,replicas=3,steps=10,seed=1,
        checkpoints=[1,10],functionals=lambda x:np.c_[x,x*x],transform=np.eye(1),max_radius=1e-20)
    assert run["stop_reason"]=="all_replicas_failed"
    assert len(run["failures"])==3
    assert not run["saved"][0]["valid"]
    assert np.isnan(run["terminal_state"]).all()


def test_schedule_consumes_only_frozen_pilot_seeds():
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    schedule=make_schedule(c)
    assert len(schedule)==90
    assert {job["seed"] for job in schedule}==set(c["seeds"]["pilot"])
    assert len({(job["seed"],job["candidate"]["id"]) for job in schedule})==90
    assert make_schedule(c)==schedule
