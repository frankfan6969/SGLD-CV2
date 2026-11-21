import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from finalize_german_p1_selection import aggregate,choose


def test_selection_requires_every_seed():
    candidate=dict(id="test",arm="full_cv2",h=.1)
    e=dict(candidates=[candidate],seeds=[1,2,3,4],budget=dict(iteration_checkpoints=[128]))
    score=dict(iteration=128,status="ok",sw2=.1,variance_nrmse=.2,
               alternative_reference_sw2=.11,alternative_reference_variance_nrmse=.21,cost=dict(total_seconds=2.))
    records=[dict(candidate=candidate,seed=s,checkpoints=[score]) for s in [1,2,3]]
    row=aggregate(records,e)[0]
    assert row["status"]=="failed_or_censored_seed_set" and row["sw2"] is None
    assert choose([row],"full_cv2",.5,.5) is None
    records.append(dict(candidate=candidate,seed=4,checkpoints=[score]))
    assert aggregate(records,e)[0]["sw2"]==.1


def test_reference_sensitivity_can_change_feasibility():
    row=dict(candidate_id="a",arm="full_cv2",h=.1,iteration=128,status="ok_all_four_seeds",
             sw2=.1,variance_nrmse=.2,alternative_reference_sw2=.3,alternative_reference_variance_nrmse=.2,total_seconds=1.)
    assert choose([row],"full_cv2",.15,None)==row
    assert choose([row],"full_cv2",.15,None,True) is None
