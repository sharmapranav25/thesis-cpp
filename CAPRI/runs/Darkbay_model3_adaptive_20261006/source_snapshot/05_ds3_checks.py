"""Behavioral checks for the frozen adaptive curriculum, without real labels."""
import copy
import numpy as np
from .ds3_data import AdaptiveSampler, percentile
from .ds2_data import BucketSampler


def run(rows):
    train=[r for r in rows if r['split']=='train']
    validation={r['index'] for r in rows if r['split']=='validation'}
    sampler=AdaptiveSampler(train,48)
    initial=list(iter(sampler))
    fixed=list(iter(BucketSampler([r['bucket'] for r in train],48)))
    assert initial==fixed, 'Initial sampling must exactly match DS2'
    assert not set(sampler.ids[initial]) & validation
    assert sampler.last_draw['dropped']==24 and sum(sampler.last_draw['optimized_draws'])==94976
    before=sampler.smooth.copy()
    reverse=np.arange(len(train),0,-1,dtype=float)/10000
    sampler.update(reverse)
    assert np.array_equal(sampler.smooth,.5*before+.5*percentile(reverse,sampler.ids))
    assert sorted(np.bincount(sampler.buckets).tolist())==[31666,31667,31667]
    for raw in [np.zeros(len(train)),np.full(len(train),1000.),reverse,reverse[::-1]]:
        old=sampler.mass.copy();list(iter(sampler));sampler.update(raw)
        assert np.max(abs(sampler.mass-old))<=.1+1e-12
        assert np.isclose(sampler.probabilities().sum(),1)
        assert sampler.probabilities().min()>0
    saved=sampler.state_dict()
    resumed=AdaptiveSampler(train,48);resumed.load_state_dict(saved)
    assert list(iter(sampler))==list(iter(resumed))
    assert sampler.last_draw==resumed.last_draw
    # Rankings depend exclusively on training records, regardless of validation contents.
    altered=copy.deepcopy(rows)
    for row in altered:
        if row['split']=='validation':row.update(nll=1e99,bucket='hard',text='unused')
    independent=AdaptiveSampler([r for r in altered if r['split']=='train'],48)
    original=AdaptiveSampler(train,48)
    assert np.array_equal(independent.smooth,original.smooth)
    invalid=reverse.copy();invalid[0]=np.nan
    try:original.update(invalid)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite scoring must fail')
    return dict(initial_sampling_exactly_matches_ds2=True,train=95000,validation=5000,
        no_validation_leakage=True,bounded_mass=True,finite_fail_closed=True,
        smoothing_verified=True,sampler_resume_exact=True,optimized_per_epoch=94976)
