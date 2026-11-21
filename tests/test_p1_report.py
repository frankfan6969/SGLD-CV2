import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_german_p1_pilot import paired_summary, pareto


def test_frontier_keeps_cost_accuracy_tradeoff():
    assert pareto([(1., 3.), (2., 2.), (3., 4.), (4., 1.)]) == [0, 1, 3]


def test_failed_seed_cannot_create_survivor_only_average():
    candidate = dict(id="example", arm="full_cv2", coefficient=.1, h=.1, alpha=None)
    config = dict(pilot_candidates=[candidate], seeds=dict(pilot=[1, 2]),
                  budgets=dict(pilot=dict(iteration_checkpoints=[128, 512])))
    good = dict(status="ok", sw2=.2, variance_nrmse=.1,
                alternative_reference_sw2=.21, alternative_reference_variance_nrmse=.11,
                cost=dict(total_seconds=1.))
    records = [dict(candidate=candidate, seed=seed, failures=[] if seed == 1 else [{}],
                    checkpoints=[dict(good, iteration=128),
                                 dict(good, iteration=512) if seed == 1 else
                                 dict(iteration=512, status="failed_ensemble")]) for seed in [1, 2]]
    _, aggregate, _ = paired_summary(records, config)
    assert aggregate[0]["status"] == "ok_both_seeds"
    assert aggregate[0]["sw2"] == .2
    assert aggregate[1]["status"] == "invalid_or_censored_seed_pair"
    assert aggregate[1]["sw2"] is None
    assert aggregate[1]["total_seconds"] is None
