"""Measured TRAIN affinity grouping and DEV-only teacher routing.

No live model is loaded or updated. No thresholds are fitted on real, calibration
or test data. Callers must freeze budgets, score definitions and guards first.
"""
import math
import re
import numpy as np
from .common import sha


def _matrix(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64)


def _rebalance(labels,affinity,minimum,maximum):
    """Keep k groups; every transfer decreases a size violation deterministically."""
    labels=labels.copy();k=int(labels.max())+1
    def gain(i,destination):
        source=labels[i];peers=np.flatnonzero((labels==source)&(np.arange(len(labels))!=i))
        old=float(affinity[i,peers].mean()) if len(peers) else 0.
        return float(affinity[i,labels==destination].mean())-old
    while True:
        counts=np.bincount(labels,minlength=k);big=np.flatnonzero(counts>maximum)
        if not len(big):break
        destinations=np.flatnonzero(counts<maximum)
        choices=[(gain(i,d),-int(i),-int(d),int(i),int(d)) for g in big
                 for i in np.flatnonzero(labels==g) for d in destinations if d!=g]
        if not choices:raise ValueError('No feasible oversized-group repair')
        *_,i,d=max(choices);labels[i]=d
    while True:
        counts=np.bincount(labels,minlength=k);small=np.flatnonzero(counts<minimum)
        if not len(small):break
        donors=np.flatnonzero(counts>minimum)
        choices=[(gain(i,d),-int(i),-int(d),int(i),int(d)) for g in donors
                 for i in np.flatnonzero(labels==g) for d in small if d!=g]
        if not choices:raise ValueError('No feasible undersized-group repair')
        *_,i,d=max(choices);labels[i]=d
    return labels


def group_measured_tasks(aggregate,candidate_counts,minimum_size,maximum_size,minimum_observations,seed):
    """Spectral grouping of symmetrized directed TAG; never substitute cosine.

    All directed pairs must have measured support. Additional TRAIN probes are
    required when evidence is missing, rather than inventing a zero affinity.
    """
    from sklearn.cluster import SpectralClustering
    from threadpoolctl import threadpool_limits
    if aggregate.get('split')!='train':raise ValueError('Grouping requires TRAIN-only affinity')
    tasks=list(aggregate['tasks']);n=len(tasks)
    if n<2 or len(set(tasks))!=n:raise ValueError('Unique measured task names required')
    if any(type(x) is not int or x<1 for x in (minimum_size,maximum_size,minimum_observations)) or maximum_size<minimum_size:
        raise ValueError('Positive explicit group size/evidence limits required')
    if not math.isfinite(float(aggregate['epsilon'])) or aggregate['epsilon']<=0:raise ValueError('Invalid TAG step')
    order=sorted(range(n),key=lambda i:tasks[i]);tasks=[tasks[i] for i in order]
    directed=_matrix(aggregate['affinity']);observations=_matrix(aggregate['affinity_counts'])
    if directed.shape!=(n,n) or observations.shape!=(n,n):raise ValueError('Affinity/evidence shape mismatch')
    directed=directed[np.ix_(order,order)];observations=observations[np.ix_(order,order)]
    if (not np.isfinite(directed).all() or not np.isfinite(observations).all()
        or not np.equal(observations,np.floor(observations)).all() or (observations<minimum_observations).any()):
        raise ValueError('Insufficient measured pair evidence; collect additional TRAIN probes')
    positive_self=float((np.diag(directed)>0).mean())
    if positive_self<.9:raise ValueError('Invalid TAG step: fewer than 90% positive self-affinities')
    similarity=(directed+directed.T)/2;off=similarity[~np.eye(n,dtype=bool)]
    scale=float(np.std(off))
    if not math.isfinite(scale):raise ValueError('Invalid affinity scale')
    scale=scale or 1.
    weights=np.exp(np.clip((similarity-float(off.max()))/scale,-50,0));np.fill_diagonal(weights,1.)
    candidates=sorted(set(candidate_counts))
    if not candidates or any(type(k) is not int or k<2 or k>=n for k in candidates):raise ValueError('Explicit candidate group counts required')
    reports=[];best=None
    for k in candidates:
        if not k*minimum_size<=n<=k*maximum_size:
            reports.append(dict(k=k,feasible=False));continue
        with threadpool_limits(limits=2):
            labels=SpectralClustering(n_clusters=k,affinity='precomputed',random_state=seed,assign_labels='discretize').fit_predict(weights)
        if len(np.unique(labels))!=k:raise ValueError('Spectral solver did not produce the requested groups')
        labels=_rebalance(labels,similarity,minimum_size,maximum_size)
        groups=sorted([sorted(tasks[i] for i in np.flatnonzero(labels==g)) for g in range(k)])
        same=labels[:,None]==labels[None,:];diagonal=np.eye(n,dtype=bool)
        within=similarity[same&~diagonal];between=similarity[~same]
        score=float(within.mean() if len(within) else 0)-float(between.mean())
        sizes=[len(group) for group in groups]
        if min(sizes)<minimum_size or max(sizes)>maximum_size:raise AssertionError('Group size repair failed')
        result=dict(k=k,feasible=True,score=score,sizes=sizes,groups=groups);reports.append(result)
        if best is None or score>best['score']:best=result
    if best is None:raise ValueError('No feasible group count for the frozen bounds')
    return dict(groups={f'g{i+1}':group for i,group in enumerate(best['groups'])},score=best['score'],
                candidates=reports,seed=seed,epsilon=aggregate['epsilon'],minimum_observations=minimum_observations,
                positive_self_fraction=positive_self,task_order=tasks,source='symmetrized directed TRAIN TAG',
                missing_evidence='refuse grouping; never zero-fill unobserved pairs')


def _hash(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def choose_teachers(baseline,candidates,policy,verify_files=True):
    """Route each assigned output only to a strictly better, guard-passing teacher.

    Reports contain metrics {name: {value, n}}, immutable checkpoint identity,
    and dev/evaluation identities. Policy contains tasks and preservation guards
    {name: {direction, max_regression}} frozen before looking at candidates.
    Lower-is-better task losses are supported explicitly, never guessed.
    """
    required=('dataset_sha256','evaluation_protocol_sha256')
    if policy.get('split')!='dev' or not all(_hash(policy.get(k)) for k in required):
        raise ValueError('Frozen DEV dataset/evaluation identity required')
    tasks=policy.get('tasks',{});guards=policy.get('guards',{})
    if not tasks or not guards:raise ValueError('Explicit task directions and preservation guards required')
    for name,direction in tasks.items():
        if direction not in ('higher','lower'):raise ValueError('Explicit metric direction required: '+name)
    for name,guard in guards.items():
        if guard.get('direction') not in ('higher','lower') or not math.isfinite(guard.get('max_regression',math.nan)) or guard['max_regression']<0:
            raise ValueError('Invalid preservation guard: '+name)
    names=set(tasks)|set(guards)
    def validate(report):
        if report.get('split')!='dev' or report.get('purpose')!='selection' or any(report.get(k)!=policy[k] for k in required):
            raise ValueError('Teacher reports must use the same declared synthetic DEV protocol')
        if (not _hash(report.get('checkpoint_sha256')) or report.get('weights') not in ('raw','ema')
            or not isinstance(report.get('checkpoint_path'),str) or not report['checkpoint_path']
            or type(report.get('epoch')) is not int or report['epoch']<1):raise ValueError('Incomplete teacher checkpoint identity')
        if verify_files and sha(report['checkpoint_path'])!=report['checkpoint_sha256']:raise ValueError('Teacher checkpoint hash mismatch')
        for name in names:
            m=report.get('metrics',{}).get(name,{})
            if not math.isfinite(m.get('value',math.nan)) or type(m.get('n')) is not int or m['n']<1:
                raise ValueError('Missing finite teacher metric/evidence: '+name)
    validate(baseline)
    decisions=[];eligible=[]
    for report in candidates:
        validate(report)
        if report['weights']!=baseline['weights']:raise ValueError('Do not compare raw and EMA selection reports')
        if any(report['metrics'][k]['n']!=baseline['metrics'][k]['n'] for k in names):
            raise ValueError('Teacher metrics have unequal evidence counts')
        assigned=report.get('assigned_tasks')
        if not assigned or len(set(assigned))!=len(assigned) or any(k not in tasks for k in assigned):
            raise ValueError('Each specialist needs a known assigned-task list')
        failures=[]
        for name,guard in guards.items():
            b=baseline['metrics'][name]['value'];v=report['metrics'][name]['value']
            regression=(b-v) if guard['direction']=='higher' else (v-b)
            if regression>guard['max_regression']:failures.append(name)
        decisions.append(dict(checkpoint_sha256=report['checkpoint_sha256'],epoch=report['epoch'],
                              eligible=not failures,failed_preservation_guards=failures))
        if not failures:eligible.append(report)
    eligible.sort(key=lambda r:(r['epoch'],r['checkpoint_sha256']))
    routes={}
    for task,direction in sorted(tasks.items()):
        selected=baseline;value=baseline['metrics'][task]['value']
        for report in eligible:
            if task not in report['assigned_tasks']:continue
            candidate=report['metrics'][task]['value']
            if (candidate>value if direction=='higher' else candidate<value):selected=report;value=candidate
        routes[task]=dict(checkpoint_path=selected['checkpoint_path'],checkpoint_sha256=selected['checkpoint_sha256'],
                          weights=selected['weights'],epoch=selected['epoch'],metric=value,
                          baseline_metric=baseline['metrics'][task]['value'],source='baseline' if selected is baseline else 'specialist')
    return dict(routes=routes,candidates=decisions,split='dev',policy=policy,
                checkpoint_files_verified=verify_files,tie_rule='baseline on equality; then earliest specialist epoch and checkpoint hash')
