import json
import zipfile
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from threadpoolctl import threadpool_limits
from sgld_cv import supplied_anchor, prepare_estimator, run_replicas
from sgld_cv.samplers import run_chains
from sgld_cv.models import GaussianRegression, LogisticRegression
from sgld_cv.preconditioners import RMSprop, metric_pilot
from sgld_cv.randomness import generator, BATCH, NOISE
from sgld_cv.diagnostics import gradient_audit, chain_diagnostics
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.references import ridge_posterior, hmc_reference
from sgld_cv.data import prepare_year, year_model
from sgld_cv.workflow import run_study, METHODS, checkpoint_within_budget
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.horizon import provisional_horizon, run_horizon_probe


def test_rmsprop_formula_state_isolation_and_invalid_inputs():
    m=RMSprop(2,alpha=.9,epsilon=.01)
    assert_allclose(m.update([2,-3]),1/(.01+np.sqrt([.4,.9])))
    a=RMSprop(2,initial_v=m.v); b=RMSprop(2,initial_v=m.v)
    a.update([100,100]);assert_allclose(b.v,m.v)
    with pytest.raises(ValueError):RMSprop(2,epsilon=0)


def test_raw_metric_gradient_is_not_cv_gradient_and_no_duplicate_evaluations():
    target=make_gaussian_target(N=30,d=2,kappa=4,seed=1)
    anchor=supplied_anchor(np.zeros(2),provenance="test")
    idx=np.array([0,1,2]);theta=np.array([.2,.3])
    raw=[]
    for name in ['sgld','cv1','cv2']:
        from sgld_cv.costs import CostLedger
        cost=CostLedger();estimator=prepare_estimator(target.model,name,anchor)
        g,r=estimator.gradient(theta,idx,cost,return_raw_mean=True)
        assert_allclose(g,estimator.gradient(theta,idx))
        assert cost.operations['observation_gradient_evaluations']==3
        raw.append(r)
    assert_allclose(raw[0],raw[1]);assert_allclose(raw[1],raw[2])
    assert_allclose(raw[0],target.model.data_gradients(theta,idx).mean(axis=0))


def test_preconditioned_transition_and_pilot_cost_once():
    target=make_gaussian_target(N=30,d=2,kappa=4,seed=1)
    anchor=supplied_anchor(np.zeros(2),provenance="test")
    estimator=prepare_estimator(target.model,'cv2',anchor)
    pilot=metric_pilot(target.model,anchor.theta,batch_size=3,batches=4)
    h=.001;initial=np.array([.1,-.2]);seed=5
    run=run_replicas(estimator,initial=initial,step_size=h,batch_size=3,steps=1,replicas=3,seed=seed,
        preconditioner='rmsprop',initial_v=pilot.second_moment,metric_cost=pilot.cost)
    for r in range(3):
        idx=generator(seed,r,BATCH).choice(30,3,replace=False)
        g,raw=estimator.gradient(initial,idx,return_raw_mean=True)
        P=RMSprop(2,initial_v=pilot.second_moment).update(raw)
        z=generator(seed,r,NOISE).normal(size=2)
        assert_allclose(run.states[-1,r],initial-h*P*g/2+np.sqrt(h*P)*z)
    assert run.costs[-1]['operations']['observation_gradient_evaluations']==30+12+9
    assert run.costs[-1]['seconds']['preconditioner_setup']==pilot.cost.seconds['preconditioner_setup']


def test_long_chain_retains_all_draws_and_final_cost_without_intermediate_snapshots():
    target=make_gaussian_target(N=30,d=2,kappa=4,seed=1)
    estimator=prepare_estimator(target.model,'cv2',supplied_anchor(np.zeros(2),provenance='test'))
    with threadpool_limits(limits=1):
        samples,chain=run_chains(estimator,initial=np.zeros((4,2)),step_size=.01,
                                  batch_size=4,warmup=7,draws=20,chains=4,seed=9)
        full=run_replicas(estimator,initial=np.zeros((4,2)),step_size=.01,
                          batch_size=4,steps=27,replicas=4,seed=9,burn_in=7,
                          checkpoints=np.arange(8,28))
    assert_array_equal(samples,full.states[full.checkpoints>7].transpose(1,0,2))
    assert len(chain.costs)==2
    assert chain.costs[-1]['operations']==full.costs[-1]['operations']
    assert chain.costs[-1]['iteration']==27


def test_frozen_metric_noise_covariance():
    model=GaussianRegression(np.zeros((2,2)),np.zeros(2))
    run=run_replicas(prepare_estimator(model,'sgld'),initial=np.zeros(2),step_size=.1,batch_size=1,
        steps=1,replicas=4096,seed=1,preconditioner='frozen',frozen_diagonal=np.array([2.,5.]))
    assert_allclose(np.cov(run.states[-1],rowvar=False),np.diag([.2,.5]),atol=.02)


def test_full_frozen_metric_uses_same_matrix_in_drift_and_noise():
    target = make_gaussian_target(N=30, d=2, kappa=4, seed=1)
    estimator = prepare_estimator(target.model, "cv2",
                                  supplied_anchor(np.zeros(2), provenance="test"))
    matrix = np.array([[2.0, 0.4], [0.4, 1.5]])
    initial = np.array([0.1, -0.2])
    h, seed = .001, 5
    run = run_replicas(estimator, initial=initial, step_size=h, batch_size=3,
                       steps=1, replicas=1, seed=seed, preconditioner="frozen_full",
                       frozen_matrix=matrix)
    idx = generator(seed, 0, BATCH).choice(30, 3, replace=False)
    z = generator(seed, 0, NOISE).normal(size=2)
    expected = initial - h * (matrix @ estimator.gradient(initial, idx)) / 2
    expected += np.sqrt(h) * (np.linalg.cholesky(matrix) @ z)
    assert_allclose(run.states[-1, 0], expected, atol=1e-15)
    assert run.costs[-1]["operations"]["dense_metric_vector_products"] == 2
    assert "full_metric_factorization" in run.costs[-1]["seconds"]
    with pytest.raises(ValueError, match="positive definite"):
        run_replicas(estimator, initial=initial, step_size=h, batch_size=3,
                     steps=1, replicas=1, seed=seed, preconditioner="frozen_full",
                     frozen_matrix=np.diag([1., -1.]))


def test_gradient_audit_preserves_mean_mse_and_quadratic_zero():
    target=make_gaussian_target(N=30,d=2,kappa=4,seed=1)
    anchor=supplied_anchor(np.zeros(2),provenance='test')
    estimators={m:prepare_estimator(target.model,m,anchor) for m in ['sgld','cv1','cv2']}
    rows,errors=gradient_audit(estimators,np.array([[.3,.2],[-.4,.1]]),batch_size=4,batches=20)
    for row in rows:assert_allclose(row['mean_mse'],errors[row['method']].mean())
    assert errors['cv2'].max()<1e-24
    assert errors['cv1'].mean()>0


def test_ess_denominators_and_adaptive_label():
    x=np.random.default_rng(2).normal(size=(4,1500,2))
    cost={'total_seconds':20.,'seconds':{'sampling':10.,'setup':10.},'operations':{'observation_gradient_evaluations':10000}}
    result=chain_diagnostics(x,cost=cost)
    assert result['reliable']
    for row in result['coordinates']:
        assert_allclose(row['ess_per_total_second'],row['ess_bulk']/20)
        assert_allclose(row['ess_per_observation_gradient'],row['ess_bulk']/10000)
        assert_allclose(row['ess_per_retained_iteration'],row['ess_bulk']/6000)
    assert not chain_diagnostics(x,adaptive=True)['reliable']
    x[0]+=5
    assert not chain_diagnostics(x)['reliable']


def test_ridge_posterior_and_hmc_gaussian_truth():
    target=make_gaussian_target(N=25,d=2,kappa=8,seed=2)
    posterior=ridge_posterior(target.model,chunk_size=7)
    assert_allclose(posterior.mean,0,atol=1e-12)
    assert_allclose(posterior.covariance,target.covariance,atol=1e-12)
    with threadpool_limits(limits=1):
        samples,diag=hmc_reference(target.model,supplied_anchor(np.zeros(2),provenance='truth'),
                      chains=4,warmup=100,draws=600,seed=19,leapfrog_steps=8)
    assert_allclose(samples.mean(axis=(0,1)),0,atol=.08)
    assert_allclose(np.cov(samples.reshape(-1,2),rowvar=False),target.covariance,atol=.09)
    assert sum(diag['divergences'])==0
    assert min(diag['acceptance'])>.5


def test_year_training_prefix_and_transform(tmp_path):
    rng=np.random.default_rng(3)
    rows=np.column_stack([np.arange(2000,2020),rng.normal(size=(20,90))])
    archive=tmp_path/'year.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('YearPredictionMSD.txt','\n'.join(','.join(str(v) for v in row) for row in rows))
    out=tmp_path/'prepared'
    meta=prepare_year(archive,out,rows=12,standardize=True)
    model,_=year_model(out)
    assert meta['validation_subset'] and model.N==12 and model.d==90
    assert isinstance(model.X,np.memmap)
    assert_allclose(model.X.mean(axis=0),0,atol=1e-12)
    assert_allclose(model.X.std(axis=0),1,atol=1e-12)
    assert_allclose(model.y,np.arange(2000,2012)-2005.5)


def tiny_config():
    return {'purpose':'pilot','problem':{'kind':'gaussian','N':20,'d':2,'kappa':3,'seed':1},
      'metric':{'batches':3,'seed':8100},'tuning':{'seeds':[100],'batch_sizes':[4],'step_fractions':[.1],
      'steps':3,'replicas':8},'evaluation':{'seeds':[300],'steps':4,'replicas':8,'reference_size':8},
      'chains':{'count':4,'warmup':5,'draws':30},'audit':{'states':2,'batches':3,'radii':[.01,.1]}}


def test_complete_pipeline_and_frozen_selection(tmp_path):
    import csv
    c=tiny_config()
    with threadpool_limits(limits=1):
        run_study(c,tmp_path/'pilot')
    out=tmp_path/'pilot'
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['reference_reliable']
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    assert {r['method'] for r in rows}==set(METHODS)
    assert all(r['status']=='ok' for r in rows)
    assert (out/'chains_pcv2_frozen_seed300.npy').exists()
    c['evaluation']['seeds']=[301]
    with threadpool_limits(limits=1):
        run_study(c,tmp_path/'heldout',frozen_selection=out/'frozen_selection.json')
    assert not (tmp_path/'heldout/tuning_trials.csv').exists()
    c['purpose']='publication'
    with pytest.raises(ValueError,match='frozen'):
        run_study(c,tmp_path/'bad')
    c['evaluation']['seeds']=[100]
    with pytest.raises(ValueError,match='disjoint'):
        run_study(c,tmp_path/'leak')


def test_common_settings_and_publication_seed_guard(tmp_path):
    c=tiny_config();c['comparison']='common'
    with threadpool_limits(limits=1):run_study(c,tmp_path/'common')
    frozen=json.loads((tmp_path/'common/frozen_selection.json').read_text())
    settings=frozen['selection']
    assert len({v['step_size'] for v in settings.values()})==1
    assert len({v['batch_size'] for v in settings.values()})==1
    c['purpose']='publication'
    with pytest.raises(ValueError,match='already inspected'):
        run_study(c,tmp_path/'reuse',frozen_selection=tmp_path/'common/frozen_selection.json')


def test_ridge_workflow_has_direct_and_cached_controls(tmp_path):
    rng=np.random.default_rng(31);prepared=tmp_path/'data';prepared.mkdir()
    np.save(prepared/'X.npy',rng.normal(size=(30,3)))
    np.save(prepared/'y.npy',rng.normal(size=30))
    (prepared/'metadata.json').write_text(json.dumps({'dataset':'test fixture','validation_subset':True}))
    c=tiny_config();c['problem']={'kind':'year_ridge','directory':str(prepared),'noise_variance':2.,'prior_precision':1.}
    c['comparison']='common'
    run_study(c,tmp_path/'ridge')
    assert (tmp_path/'ridge/ridge_control_metrics.csv').exists()
    controls=json.loads((tmp_path/'ridge/ridge_controls.json').read_text())
    assert controls['direct_total_seconds']>=controls['direct_sampling_seconds']
    assert controls['cached_ula_costs'][-1]['operations']['dense_precision_vector_products']==8*4


def test_standardized_sliced_w2_fixed_calibration_and_unequal_sizes():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(48,3))
    shift=np.array([2.,0.,0.])
    metric=make_accuracy_metric(3,kind='standardized_sliced_w2',center=np.zeros(3),
                                covariance=np.diag([4.,1.,1.]),projections=4096,seed=9)
    assert_allclose(metric.distance(x,x+shift),1/np.sqrt(3),rtol=.03)
    assert metric.distance(x,x)<1e-12
    assert np.isfinite(metric.distance(x[:17],x))
    same=make_accuracy_metric(3,kind='standardized_sliced_w2',center=np.zeros(3),
                              covariance=np.diag([4.,1.,1.]),projections=4096,seed=9)
    assert_allclose(metric.projections,same.projections)
    with pytest.raises(ValueError):metric.distance(x,np.full_like(x,np.nan))
    assert make_accuracy_metric(2,kind='auto').kind=='w2'


def test_provisional_horizon_requires_sustained_stability():
    rows=[]
    for k,value in zip((0,10,20,40,80,160), (3,2,1.02,1.01,1.0,1.01)):
        rows.append({'iteration':k,'accuracy':value,'mean_error_std':value,
                     'covariance_error_std':value,'status':'ok'})
    assert provisional_horizon(rows,tolerance=.05,window=3)==20
    rows[-1]['accuracy']=2
    assert provisional_horizon(rows,tolerance=.05,window=3) is None


def test_common_wall_budget_scores_last_eligible_checkpoint():
    ks=np.array([0,10,20,30])
    costs=[{'total_seconds':v} for v in (.2,.5,.9,1.3)]
    assert checkpoint_within_budget(ks,costs,1.0)==2
    assert checkpoint_within_budget(ks,costs,.1) is None
    assert checkpoint_within_budget(ks,costs,2.0) is None
    assert checkpoint_within_budget(ks,costs,None)==3


def test_logistic_horizon_probe_saves_curves_and_provisional_flags(tmp_path):
    c=tiny_config()
    c['problem']={'kind':'synthetic_logistic','N':30,'d':2,'design_kappa':3,
                  'seed':1,'prior_precision':1.}
    c['reference']={'chains':4,'warmup':60,'draws':120,'seed':7000,
                    'step_size':.3,'leapfrog_steps':8}
    c['horizon_probe']={'seeds':[120],'batch_size':4,'step_fraction':.1,
                        'max_steps':4,'checkpoints':[0,1,2,3,4],
                        'replicas':8,'reference_size':8,'plateau_window':3}
    with threadpool_limits(limits=1): result=run_horizon_probe(c,tmp_path/'probe')
    assert result['phase']=='rough' and set(result['provisional_k_by_method'])==set(METHODS)
    assert (tmp_path/'probe/horizon_curves.csv').exists()
    assert (tmp_path/'probe/probe_pcv2_seed120.npz').exists()
