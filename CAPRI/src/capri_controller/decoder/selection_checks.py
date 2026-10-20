"""CPU-only checks for measured grouping and teacher-selection boundaries."""
import copy
import time
from pathlib import Path
import numpy as np
from .common import write,sha,now
from .specialist_selection import group_measured_tasks,choose_teachers,_rebalance


def check_selection(run):
    names=[f'task_{i:02d}' for i in range(12)];affinity=np.full((12,12),-.01)
    affinity[:6,:6]=.04;affinity[6:,6:]=.04;np.fill_diagonal(affinity,.03)
    aggregate=dict(tasks=names,affinity=affinity,affinity_counts=np.full((12,12),3),epsilon=.003,split='train')
    grouped=group_measured_tasks(aggregate,[2,3],3,8,3,731)
    expected={frozenset(names[:6]),frozenset(names[6:])}
    assert {frozenset(x) for x in grouped['groups'].values()}==expected
    shuffled=copy.deepcopy(aggregate);permutation=np.arange(12)[::-1]
    shuffled['tasks']=[names[i] for i in permutation]
    for key in ('affinity','affinity_counts'):shuffled[key]=shuffled[key][np.ix_(permutation,permutation)]
    assert group_measured_tasks(shuffled,[3,2],3,8,3,731)==grouped
    repaired=_rebalance(np.array([0]*10+[1]*2),affinity,5,7)
    assert min(np.bincount(repaired))>=5 and max(np.bincount(repaired))<=7
    checks=dict(known_groups_recovered=True,task_and_candidate_order_invariant=True,group_size_bounds_repaired=True)
    for label,change in [('missing_pair',lambda a:a['affinity_counts'].__setitem__((0,1),0)),
                         ('bad_self_step',lambda a:a['affinity'].__setitem__(([0,1],[0,1]),-.01)),
                         ('heldout',lambda a:a.__setitem__('split','test'))]:
        broken=copy.deepcopy(aggregate);change(broken)
        try:group_measured_tasks(broken,[2],3,8,3,731)
        except ValueError:pass
        else:raise AssertionError('Unsafe grouping accepted: '+label)
    try:group_measured_tasks(aggregate,[5],3,8,3,731)
    except ValueError:pass
    else:raise AssertionError('Infeasible group limits accepted')
    checks['missing_pair_bad_step_heldout_and_infeasible_grouping_refused']=True
    root=Path(run)/('teacher_selection_smoke_'+str(time.time_ns()));root.mkdir()
    policy=dict(split='dev',dataset_sha256='a'*64,evaluation_protocol_sha256='b'*64,
                tasks={'presence/blur':'higher','font_contrastive':'lower'},
                guards={k:dict(direction='higher',max_regression=0.) for k in ('count_accuracy','localization_f1','font_top1')})
    def report(name,epoch=1,**values):
        path=root/(name+'.bin');path.write_bytes(name.encode())
        scores=dict({'presence/blur':.5,'font_contrastive':.7,'count_accuracy':.9,'localization_f1':.8,'font_top1':.6},**values)
        return dict(split='dev',purpose='selection',dataset_sha256='a'*64,evaluation_protocol_sha256='b'*64,
                    checkpoint_path=str(path),checkpoint_sha256=sha(path),epoch=epoch,weights='ema',
                    assigned_tasks=list(policy['tasks']),metrics={k:dict(value=v,n=100) for k,v in scores.items()})
    base=report('baseline')
    improved=report('improved',2,**{'presence/blur':.6,'font_contrastive':.6})
    weak=report('weak',1,**{'presence/blur':.4,'font_contrastive':.8})
    unsafe=report('count_regression',1,**{'presence/blur':.99,'font_contrastive':.01,'count_accuracy':.89})
    tie=report('equal',1)
    route=choose_teachers(base,[unsafe,weak,improved,tie],policy)
    assert all(x['checkpoint_sha256']==improved['checkpoint_sha256'] for x in route['routes'].values())
    assert route['candidates'][0]['failed_preservation_guards']==['count_accuracy']
    equal=choose_teachers(base,[tie,weak],policy)
    assert all(x['source']=='baseline' for x in equal['routes'].values())
    only_blur=copy.deepcopy(improved);only_blur['assigned_tasks']=['presence/blur']
    restricted=choose_teachers(base,[only_blur],policy)
    assert restricted['routes']['font_contrastive']['source']=='baseline'
    assert restricted['routes']['presence/blur']['source']=='specialist'
    later=report('later_same_score',3,**{'presence/blur':.6,'font_contrastive':.6})
    for candidates in ([later,improved],[improved,later]):
        chosen=choose_teachers(base,candidates,policy)
        assert all(x['epoch']==2 for x in chosen['routes'].values())
    checks.update(strict_task_improvement_required=True,baseline_wins_equal_or_worse=True,
                  preservation_regression_rejects_teacher=True,unassigned_routes_stay_baseline=True,
                  deterministic_specialist_ties=True)
    variants=[]
    for field,value in [('split','test'),('purpose','optimization'),('dataset_sha256','c'*64),('weights','raw')]:
        bad=copy.deepcopy(improved);bad[field]=value;variants.append(bad)
    bad=copy.deepcopy(improved);bad['metrics']['font_contrastive']['n']=99;variants.append(bad)
    bad=copy.deepcopy(improved);bad['metrics']['presence/blur']['value']=float('nan');variants.append(bad)
    for bad in variants:
        try:choose_teachers(base,[bad],policy)
        except ValueError:pass
        else:raise AssertionError('Incompatible teacher report accepted')
    Path(improved['checkpoint_path']).write_bytes(b'tampered')
    try:choose_teachers(base,[improved],policy)
    except ValueError:pass
    else:raise AssertionError('Teacher hash mismatch accepted')
    checks['heldout_mismatched_evidence_nonfinite_and_tampered_teachers_refused']=True
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,analytic_groups=grouped,
                source_sha256=sha(Path(__file__).with_name('specialist_selection.py')),
                limitation='Fabricated affinity/metric fixtures only; no actual CAPRI task groups or teachers selected. Real grouping awaits a trained joint baseline and adequate TRAIN pair evidence. Real teacher routing requires frozen DEV metrics/guards.')
    write(Path(run)/'specialist_selection_checks.json',result)
    return result
