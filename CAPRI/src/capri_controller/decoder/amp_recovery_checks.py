"""Fault-injected AMP checks before resuming the interrupted real phase."""
import copy
from pathlib import Path
import torch
from .common import read,write,sha,now
from .checkpoints import rng_state,restore_rng
from .phase_engine_checks import equal
from .amp_recovery import safe_step,MAX_ATTEMPTS


class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.layer=torch.nn.Linear(4,7);self.dropout=torch.nn.Dropout(.2)
    def forward(self,pixels,valid_pixels):return dict(count_logits=self.layer(self.dropout(pixels)))


def check_amp_recovery(run):
    from . import training
    if not torch.cuda.is_available():raise RuntimeError('CUDA needed to verify actual AMP recovery')
    training.initialize();checks={}
    def objective(pred,targets,weights):
        loss=pred['count_logits'].float().square().mean();return loss,dict(toy=loss)
    def setup(device,scale):
        m=Tiny().to(device);o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False)
        s=torch.optim.lr_scheduler.StepLR(o,1,gamma=.9);sc=torch.amp.GradScaler(device,init_scale=scale,growth_interval=2000)
        return m,o,s,sc,copy.deepcopy(m.state_dict())
    batch=dict(inputs=dict(pixels=torch.arange(12).reshape(3,4).float()/12,valid_pixels=torch.ones(3,4,dtype=torch.bool)),
               targets=[dict(count=2)]*3,audit=[dict(stem='toy_'+str(i)) for i in range(3)])
    for device in ('cpu','cuda'):
        initial=setup(device,8.);reference=copy.deepcopy(initial[0].state_dict());rng=rng_state()
        good=safe_step(*initial,batch,{},1,objective,training.update_ema)
        expected=[copy.deepcopy(initial[0].state_dict()),copy.deepcopy(initial[1].state_dict()),
                  copy.deepcopy(initial[2].state_dict()),copy.deepcopy(initial[3].state_dict()),copy.deepcopy(initial[4])]
        after_rng=rng_state();retry=setup(device,16.);retry[0].load_state_dict(reference);retry[4].update(copy.deepcopy(reference))
        fired=[]
        def inject(grad):
            if not fired:fired.append(True);return torch.full_like(grad,float('inf'))
            return grad
        hook=retry[0].layer.weight.register_hook(inject);restore_rng(rng);events=[]
        actual=safe_step(*retry,batch,{},1,objective,training.update_ema,events.append);hook.remove()
        found=[retry[0].state_dict(),retry[1].state_dict(),retry[2].state_dict(),retry[3].state_dict(),retry[4]]
        assert all(equal(a,b) for a,b in zip(expected,found)) and equal(after_rng,rng_state())
        assert good['amp_retries']==0 and actual['amp_retries']==1 and len(events)==1
        assert events[0]['retry_scale']==8. and events[0]['same_batch']
        checks[device+'_retry_matches_clean_raw_ema_optimizer_scheduler_scaler_rng']=True
        broken=setup(device,512.);before=copy.deepcopy(broken[0].state_dict());before_ema=copy.deepcopy(broken[4]);last=broken[2].last_epoch
        hook=broken[0].layer.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')));events=[]
        try:safe_step(*broken,batch,{},1,objective,training.update_ema,events.append)
        except FloatingPointError as exc:assert 'Persistent' in str(exc)
        else:raise AssertionError('Persistent nonfinite gradients accepted')
        hook.remove()
        assert equal(before,broken[0].state_dict()) and equal(before_ema,broken[4])
        assert not broken[1].state and broken[2].last_epoch==last and len(events)==MAX_ATTEMPTS-1
        checks[device+'_persistent_failure_bounded_without_update']=True
    run=Path(run);root=run/'amp_recovery';root.mkdir(exist_ok=True)
    if (root/'amendment.json').exists():raise FileExistsError('Preserve established AMP recovery amendment')
    protocol=read(run/'training/baseline_localization/protocol.json')
    if protocol['sources']!=training.sources():raise ValueError('Original training sources changed')
    latest=read(run/'training_recovery/baseline_localization/latest.json')
    if latest['epoch']!=10000 or sha(latest['path'])!=latest['sha256']:raise ValueError('Expected verified update-10000 recovery')
    result=dict(created_at=now(),passed=True,checks=checks,original_sources=training.sources(),
                original_protocol_sha256=sha(run/'training/baseline_localization/protocol.json'),
                recovery_source_sha256=sha(Path(__file__).with_name('amp_recovery.py')),
                check_source_sha256=sha(__file__),starting_update=10000,starting_checkpoint=latest,
                max_attempts=MAX_ATTEMPTS,policy='Retry same batch/RNG with half loss scale, reset scale growth tracker; preserve optimizer/scheduler/EMA/sample cursor until success',
                limitation='Tiny CPU/CUDA AMP fixtures prove bounded retry state semantics. Actual CAPRI overflow cause and successful replay remain to be observed during resume.')
    write(root/'amendment.json',result);return result
