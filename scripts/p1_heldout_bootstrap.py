"""Exact weighted-replica SW2 bootstrap, conditional on fixed HMC references.

Pre-sorting projections avoids sorting every resampled cloud. Integer bootstrap
weights exactly reproduce duplicating whole replicas; reference rows stay fixed.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def weighted_sw2(sorted_x,order,prefix_y,prefix_y2,counts):
    R,P=sorted_x.shape;S=prefix_y.shape[1]-1
    if S%R:raise ValueError("reference size must be integer multiple of replica count")
    factor=S//R;total=np.zeros(prefix_y.shape[0])
    for p in range(P):
        lo=0
        for j in range(R):
            n=factor*counts[order[j,p]]
            if n:
                hi=lo+n;x=sorted_x[j,p]
                for ref in range(len(total)):
                    total[ref]+=n*x*x-2*x*(prefix_y[ref,hi,p]-prefix_y[ref,lo,p])+prefix_y2[ref,hi,p]-prefix_y2[ref,lo,p]
                lo=hi
    return np.sqrt(np.maximum(total/(S*P),0))


def prepare_projected(x,references):
    order=np.argsort(x,axis=0)
    sx=np.take_along_axis(x,order,axis=0)
    y=np.sort(np.asarray(references),axis=1)
    zero=np.zeros((len(y),1,y.shape[2]))
    return sx,order,np.concatenate([zero,np.cumsum(y,axis=1)],axis=1),np.concatenate([zero,np.cumsum(y*y,axis=1)],axis=1)


def bootstrap_design(seeds,replicas,repeats,seed):
    rng=np.random.default_rng(seed)
    outer=rng.integers(0,seeds,size=(repeats,seeds))
    counts=np.empty((repeats,seeds,replicas),dtype=np.int16)
    for b in range(repeats):
        for slot in range(seeds):counts[b,slot]=np.bincount(rng.integers(0,replicas,replicas),minlength=replicas)
    return outer,counts


def interval(draws,observed,level):
    low,high=np.quantile(draws,[(1-level)/2,(1+level)/2],axis=0)
    return dict(estimate=np.asarray(observed).tolist(),lower=low.tolist(),upper=high.tolist(),level=level,
        scope="hierarchical paired percentile interval; conditional on frozen correlated reference pools; not guaranteed population-W2 coverage")
