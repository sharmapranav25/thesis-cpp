"""Small analytic TAG probes; no CAPRI model, optimizer step or learned grouping."""
import random
from pathlib import Path
import numpy as np
import torch
from .affinity import tag_batch,aggregate_tag,choose_epsilon,assigned_objective
from .checkpoints import rng_state
from .common import write,sha,now


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared=torch.nn.Parameter(torch.ones(2));self.private=torch.nn.Parameter(torch.tensor(.5))
        self.register_buffer('calls',torch.zeros((),dtype=torch.long));self.dropout=torch.nn.Dropout(.5)
    def forward(self,x):
        self.calls.add_(1)
        # Exercise RNG preservation and functional buffer isolation explicitly.
        noise=torch.rand((),device=x.device)+random.random()+float(np.random.random())
        return dict(shared=self.shared*x+noise*0,private=self.private)


def losses(output):
    x=output['shared'];p=output['private']
    values=dict(a=x[0].square(),aligned=(x[0]+1).square(),opposed=(x[0]-2).square(),
                orthogonal=x[1].square(),unlabelled=x[1].square(),private=(p-1).square())
    return values,{k:0 if k=='unlabelled' else 1 for k in values}


def same_rng(a,b):
    assert a['python']==b['python']
    assert a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
    assert torch.equal(a['torch'],b['torch'])
    assert len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))


def check_affinity(run):
    torch.set_num_threads(2);torch.manual_seed(715);random.seed(715);np.random.seed(715)
    model=Probe();model.train();model.dropout.eval();model.shared.grad=torch.tensor([.7,-.2])
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    modes=[m.training for m in model.modules()];gradient=model.shared.grad.clone();rng=rng_state()
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    tasks=['a','aligned','opposed','orthogonal','unlabelled','private']
    measured=tag_batch(model,(torch.ones(2),),losses,tasks,['shared'],.1,'train')
    assert measured['affinity'][0,0]>0 and measured['affinity'][0,1]>0 and measured['affinity'][0,2]<0
    assert measured['affinity'][0,3]==0
    assert torch.allclose(measured['cosine'][:4,:4],torch.tensor([[1.,1.,-1.,0.],[1.,1.,-1.,0.],[-1.,-1.,1.,0.],[0.,0.,0.,1.]]))
    assert measured['ineligible']=={'unlabelled':'no_valid_labels','private':'no_shared_gradient'}
    assert torch.isnan(measured['affinity'][4:]).all() and torch.isnan(measured['affinity'][:,4:]).all()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert modes==[m.training for m in model.modules()] and torch.equal(gradient,model.shared.grad)
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32);same_rng(rng,rng_state())
    again=tag_batch(model,(torch.ones(2),),losses,tasks,['shared'],.1,'train')
    assert torch.allclose(measured['affinity'],again['affinity'],equal_nan=True)
    summary=aggregate_tag([measured,again])
    assert (summary['affinity_counts'][:4,:4]==2).all() and (summary['affinity_counts'][4:]==0).all()
    try:choose_epsilon([summary])
    except ValueError:pass
    else:raise AssertionError('Grouping allowed with unrepresented tasks')
    good_tasks=tasks[:4]
    large=aggregate_tag([tag_batch(model,(torch.ones(2),),losses,good_tasks,['shared'],5.,'train')])
    small=aggregate_tag([tag_batch(model,(torch.ones(2),),losses,good_tasks,['shared'],.1,'train')])
    assert large['positive_self_fraction']==0 and small['positive_self_fraction']==1
    assert choose_epsilon([large,small])['epsilon']==.1
    try:choose_epsilon([large])
    except ValueError:pass
    else:raise AssertionError('Invalid lookahead accepted')
    calls=[0]
    def fail_after_base(output):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('injected lookahead failure')
        return losses(output)
    rng=rng_state()
    try:tag_batch(model,(torch.ones(2),),fail_after_base,good_tasks,['shared'],.1,'train')
    except RuntimeError as exc:assert 'injected' in str(exc)
    else:raise AssertionError('Failure was not injected')
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert modes==[m.training for m in model.modules()] and torch.equal(gradient,model.shared.grad)
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32);same_rng(rng,rng_state())
    try:tag_batch(model,(torch.ones(2),),losses,good_tasks,['shared'],.1,'dev')
    except ValueError:pass
    else:raise AssertionError('Non-train affinity accepted')
    a=torch.tensor(2.,requires_grad=True);b=torch.tensor(3.,requires_grad=True);anchor=torch.tensor(1.,requires_grad=True)
    unknown=torch.tensor(4.,requires_grad=True)
    objective=assigned_objective({'a':a.square(),'b':b.square(),'count':anchor.square(),'unknown':unknown.square()},
        {'a':1,'b':1,'count':1,'unknown':0},['a','unknown'],['count'],
        {'a':'presence','b':'presence','count':'preservation','unknown':'presence'},{'presence':1.,'preservation':.5})
    assert objective==4.5
    objective.backward()
    assert a.grad==4 and b.grad is None and anchor.grad==1 and unknown.grad is None
    cuda=False
    if torch.cuda.is_available():
        gpu=Probe().cuda();gpu.train();gpu.dropout.eval()
        state={k:v.detach().clone() for k,v in gpu.state_dict().items()}
        measured_gpu=tag_batch(gpu,(torch.ones(2,device='cuda'),),losses,good_tasks,['shared'],.1,'train')
        assert torch.allclose(measured_gpu['affinity'],small['affinity'],atol=1e-6)
        assert all(torch.equal(state[k],v) for k,v in gpu.state_dict().items());cuda=True
    result=dict(created_at=now(),passed=True,checks=dict(directed_positive_negative_orthogonal_transfer=True,
                companion_gradient_cosine=True,missing_evidence_remains_nan=True,observation_counts_preserved=True,
                invalid_step_backtracking_gate=True,unrepresented_task_refusal=True,
                parameters_buffers_existing_gradients_rng_modes_tf32_unchanged=True,restoration_after_exception=True,
                synthetic_train_only=True,unassigned_and_unlabelled_gradients_excluded=True,cuda_agrees_with_cpu=cuda),
                source_sha256=sha(Path(__file__).with_name('affinity.py')),
                limitation='Analytic two-parameter probe only. No CAPRI baseline measured, no optimizer/EMA updated, no real grouping or specialist training performed. Declare actual shared CNN parameters/task losses and measure the trained baseline before clustering.')
    write(run/'affinity_checks.json',result)
    return result
