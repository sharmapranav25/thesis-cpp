"""Full CPU joint-phase continuation checks with tiny synthetic fixtures."""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import now,write,sha
from .specialist_checks import extended_fixture
from .specialist_tasks import task_manifest,COUNT_TASKS,LOCAL_TASKS,task_losses,specialist_objective
from .font_extraction_checks import BOXES
from .reference_sampling import ReferenceSampler
from .joint_phase import JointPreparation,JointObjective
from .phase_engine import run_phase
from .phase_engine_checks import equal
from .objectives import instance_losses


class JointScenes:
    snapshot_hash='cpu-joint-scenes-eight-v1'
    rows=[dict(stem=f'toy_{i:07d}',shape=(16,32)) for i in range(8)]
    def __init__(self):self.pred,self.targets=extended_fixture()
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        t=copy.deepcopy(self.targets[i%2])
        return dict(inputs=dict(pixels=torch.full((3,16,32),float(i%2)),valid_pixels=torch.ones(16,32,dtype=torch.bool)),
            targets=t,audit=dict(stem=self.rows[i]['stem'],font_keys=['font-'+str(j) for j in t['font_family'].tolist()]))


class JointReferences:
    snapshot_hash='cpu-joint-six-references-v1'
    def __init__(self):
        self.rows=[dict(record=dict(split='train',specimen_id='reference/'+str(i),family_group=('alpha','beta','gamma')[i//2],
            font_key='font-'+str(i//2),font_sha256=str(i//2+1)*64,style='regular',axes={}),
            shape=(16,32 if i%2==0 else 48)) for i in range(6)]
    def __getitem__(self,i):
        row=self.rows[i];h,w=row['shape']
        return dict(inputs=dict(pixels=torch.full((3,h,w),float(i+2)),valid_pixels=torch.ones(h,w,dtype=torch.bool)),
            library_labels=dict(row['record'],boxes_visible=BOXES[:2].clone()),
            audit=dict(stem=row['record']['specimen_id'],pixel_sha256=str(i+1)*64))


class JointModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.pred,_=extended_fixture();self.scene_font=torch.nn.Parameter(torch.randn(2,7,128))
        self.reference_font=torch.nn.Parameter(torch.randn(6,4,128));self.dropout=torch.nn.Dropout(.1)
    def resume_configuration(self):return dict(diagnostic='joint-model-cpu-v1')
    def forward(self,pixels,valid_pixels):
        ids=pixels[:,0,0,0].long();n=len(ids)
        if (ids<2).all():
            pred={k:({name:t[ids] for name,t in v.items()} if isinstance(v,dict) else v[ids]) for k,v in self.pred.items()}
            pred['font_embeddings']=self.dropout(self.scene_font[ids])
            pred['auxiliary']=[{k:pred[k] for k in ('presence_logits','boxes','mask_logits','count_logits')} for _ in range(3)]
            return pred
        if (ids<2).any():raise ValueError('Fixture scene/reference mixing')
        return dict(font_embeddings=self.dropout(self.reference_font[ids-2]),boxes=BOXES[None].repeat(n,1,1),
            presence_logits=torch.full((n,4),8.),visibility_logits=torch.tensor([[8.,8.,-8.,-8.]]).repeat(n,1))


class TrackedPreparation(JointPreparation):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.trace=[]
    def __call__(self,batch,context):
        prepared=super().__call__(batch,context)
        self.trace.append(dict(context,selection=copy.deepcopy(prepared['reference_selection'])))
        return prepared


def check_joint_phase(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for joint phase diagnostics')
    torch.set_num_threads(2);root=Path(run)/('joint_phase_smoke_'+str(time.time_ns()));checks={}
    mapping=dict(alpha=0,beta=1,gamma=2);dataset=JointScenes()
    tasks=task_manifest();groups={v:1. for v in tasks.values()};scales={k:1. for k in tasks}
    aux={k:1. for k in COUNT_TASKS+LOCAL_TASKS if k!='visibility'}
    policy=dict(instance_threshold=.5,visibility_threshold=.5,library_iou_threshold=.5)
    def construct(seed=312,scale=16.):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=JointModel();opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,3,.8);scaler=torch.amp.GradScaler('cpu',init_scale=scale,growth_interval=2)
        sampler=ReferenceSampler(JointReferences(),mapping,713)
        prep=TrackedPreparation(sampler,every_updates=2,max_references=2,mode='family')
        objective=JointObjective(mapping,list(tasks),groups,scales,policy,.1,.25,aux)
        return model,opt,scheduler,scaler,prep,objective
    def evaluate(model,epoch):
        assert not model.training and not torch.is_grad_enabled()
        return dict(diagnostic_epoch=epoch,mean_weight=float(model.scene_font.mean()))
    def execute(name,parts,**changes):
        options=dict(run=root/name,phase='joint_toy',model=parts[0],dataset=dataset,optimizer=parts[1],scheduler=parts[2],
            scaler=parts[3],batch_preparer=parts[4],compute_loss=parts[5],evaluate=evaluate,
            protocol=dict(diagnostic='joint-reference-engine-v1',scheduler='StepLR-3-0.8'),
            epochs=2,batch_size=2,seed=731,workers=0,recovery_every=2,diagnostic=True)
        options.update(changes);return run_phase(**options)
    def state(name,epoch=2):
        return torch.load(root/name/'phase_engine_diagnostics/joint_toy'/f'epoch_{epoch:04d}'/'checkpoint.pt',
                          map_location='cpu',weights_only=False)
    fields=('raw','ema','optimizer','scheduler','scaler','sampler_state','rng','execution','metrics','optimizer_layout')
    full=construct();execute('full',full);before=construct()
    try:execute('resume',before,interrupt_after=2)
    except RuntimeError as exc:assert 'Injected' in str(exc)
    else:raise AssertionError('Expected diagnostic interruption')
    after=construct(919);execute('resume',after)
    assert full[4].trace==before[4].trace+after[4].trace
    assert all(equal(state('full')[k],state('resume')[k]) for k in fields)
    assert all((row['selection'] is not None)==((row['optimizer_update']-1)%2==0) for row in full[4].trace)
    assert {r['epoch'] for r in full[4].trace}=={1,2}
    checks['complete_joint_raw_ema_optimizer_scheduler_scaler_rng_resume_exact']=True
    checks['epoch_and_global_step_reference_schedule_reproduced']=True
    # An overflow on the first (reference-bearing) update must reuse its selected
    # pixels and restore dropout RNG, with no second preparation call.
    low=construct(scale=8.);execute('low',low)
    retry=construct();fired=[]
    def overflow_once(grad):
        if not fired:fired.append(True);return torch.full_like(grad,float('inf'))
        return grad
    hook=retry[0].reference_font.register_hook(overflow_once)
    execute('retry',retry);hook.remove()
    assert fired and low[4].trace==retry[4].trace
    assert all(equal(state('low')[k],state('retry')[k]) for k in fields)
    checks['reference_bearing_amp_retry_exact_with_single_preparation']=True
    # Verify the no-reference path and explicit auxiliary localization coefficient.
    parts=construct();model=parts[0];model.eval()
    from .data import exact_shape_collate
    batch=exact_shape_collate([dataset[0],dataset[1]])
    prepared=parts[4](batch,dict(epoch=1,optimizer_update=2,consumed_batches=1))
    assert not prepared['reference_batches']
    actual=parts[5](model,prepared);pred=model(**batch['inputs']);losses,support,_=task_losses(pred,batch['targets'])
    expected=specialist_objective(losses,support,list(tasks),(),groups,scales)
    for auxiliary in pred['auxiliary']:
        terms,_=instance_losses(auxiliary,batch['targets']);expected+=.25*sum(terms[k]*aux[k] for k in terms)
    torch.testing.assert_close(actual,expected)
    checks['scene_only_schedule_preserves_joint_and_auxiliary_supervision']=True
    hashes={epoch:sha(root/'full/phase_engine_diagnostics/joint_toy'/f'epoch_{epoch:04d}'/'checkpoint.pt') for epoch in (1,2)}
    execute('full',construct())
    assert hashes=={epoch:sha(root/'full/phase_engine_diagnostics/joint_toy'/f'epoch_{epoch:04d}'/'checkpoint.pt') for epoch in (1,2)}
    checks['all_epoch_states_retained_and_completed_rerun_idempotent']=True
    bad=construct();bad[4].contract['every_updates']=3
    try:execute('full',bad)
    except ValueError as exc:assert 'protocol mutation' in str(exc)
    else:raise AssertionError('Changed reference schedule accepted')
    bad=construct();bad[5].contract['temperature']=.2
    try:execute('full',bad)
    except ValueError as exc:assert 'protocol mutation' in str(exc)
    else:raise AssertionError('Changed font objective accepted')
    checks['changed_reference_schedule_and_objective_refused_on_resume']=True
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,root=str(root),
        joint_source_sha256=sha(Path(__file__).with_name('joint_phase.py')),
        engine_sha256=sha(Path(__file__).with_name('phase_engine.py')),
        limitation='Tiny CPU joint integration with fabricated predictions, not actual CAPRI training/DEV quality. GPU memory/throughput readiness, production budget/weights, initialization and held-out font-gallery evaluation remain required.')
    write(Path(run)/'joint_phase_checks.json',result);return result
