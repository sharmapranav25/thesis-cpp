"""Analytic DEV metric/evaluation checks without reading actual held-out data."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from .common import now,write,sha
from .specialist_checks import extended_fixture
from .specialist_tasks import task_losses,task_manifest
from .data import EFFECT_HEADS
from .joint_evaluation import JointMetrics,contract,evaluate_joint
from .checkpoints import rng_state,execution_state
from .phase_engine_checks import equal


class ToyDev:
    split='dev';snapshot_hash='cpu-only-toy-dev-snapshot'
    rows=[dict(stem=f'toy_{i:07d}',shape=(64,128)) for i in range(2)]
    def __init__(self):self.pred,self.targets=extended_fixture()
    def __len__(self):return 2
    def __getitem__(self,i):
        return dict(inputs=dict(pixels=torch.full((3,64,128),float(i)),valid_pixels=torch.ones(64,128,dtype=torch.bool)),
                    targets=copy.deepcopy(self.targets[i]),audit=dict(stem=self.rows[i]['stem']))


class ToyModel(torch.nn.Module):
    def __init__(self,pred):super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.));self.pred=pred
    def forward(self,pixels,valid_pixels):
        assert not self.training and not torch.is_grad_enabled()
        random.random();np.random.rand();torch.rand(1)
        ids=pixels[:,0,0,0].long()
        return {k:{a:b[ids] for a,b in v.items()} if isinstance(v,dict) else v[ids] for k,v in self.pred.items()}


def check_joint_evaluation(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for joint evaluation diagnostics')
    torch.set_num_threads(2);pred,targets=extended_fixture();meter=JointMetrics();meter.add(pred,targets)
    report=meter.finish(2);losses,support,_=task_losses(pred,targets)
    for name in task_manifest():
        metric=report['metrics'][name];assert metric['n']==support[name]
        assert metric['value']==float(losses[name]) if support[name] else metric['value'] is None
    assert report['metrics']['count_accuracy']==dict(value=1.,n=2,direction='higher')
    assert report['metrics']['localization_f1']==dict(value=1.,n=5,direction='higher')
    assert report['count']['correct']==2 and report['localization']['false_positive']==0
    active='intensity/'+EFFECT_HEADS[0];inactive='intensity/'+EFFECT_HEADS[1]
    assert report['metrics'][active+'/mae']['value']==0. and report['metrics'][active+'/mae']['n']==1
    assert report['metrics'][inactive+'/mae']['value'] is None and report['metrics'][inactive]['n']==0
    assert report['metrics']['presence/'+EFFECT_HEADS[0]+'/balanced_accuracy']['value']==1.
    assert report['metrics']['presence/'+EFFECT_HEADS[1]+'/balanced_accuracy']['value'] is None
    assert report['metrics']['font_x_height']['n']==3 and report['metrics']['font_contrastive']['n']==4
    assert 'font_top1' not in report['metrics']
    checks=dict(all_96_loss_terms_preserve_support_units=True,count_and_localization_exact=True,
                conditional_intensity_masks_absence=True,single_class_balanced_accuracy_undefined=True,
                unknown_font_attributes_masked=True,no_retrieval_accuracy_from_contrastive_loss=True)
    # Different valid-evidence counts must use supported means, not batch means.
    second=copy.deepcopy(targets);second[0]['intensity_valid'][0]=False;second[0]['effect_presence'][0]=0.
    second[1]['effect_presence'][0]=1.;second[1]['intensity_valid'][0]=True;second[1]['effect_intensity'][0]=.8
    meter.add(pred,second);twice=meter.finish(4)
    assert abs(twice['metrics'][active+'/mae']['value']-.2)<1e-6 and twice['metrics'][active+'/mae']['n']==2
    second[0]['effect_presence'][0]=1.;second[0]['intensity_valid'][0]=True;second[0]['effect_intensity'][0]=1.
    meter.add(pred,second);three=meter.finish(6)
    expected=(0.+.4+.6+.4)/4
    assert abs(three['metrics'][active+'/mae']['value']-expected)<1e-6
    checks['unequal_support_weighting_correct']=True
    # Background padding is excluded; partial coverage is unpremultiplied first.
    modified=copy.deepcopy(pred)
    for i,t in enumerate(targets):modified['background_under'][i,:,t['valid_mask']==0]=999.
    other=JointMetrics();other.add(modified,targets)
    assert other.finish(2)['metrics']['background_under/mae']==report['metrics']['background_under/mae']
    partial=copy.deepcopy(targets)
    for t in partial:t['valid_mask']*=.5;t['background_under']*=.5
    other=JointMetrics();other.add(pred,partial)
    assert other.finish(2)['metrics']['background_under/mae']==report['metrics']['background_under/mae']
    checks['background_padding_and_partial_coverage_correct']=True
    dataset=ToyDev();model=ToyModel(dataset.pred);frozen=contract(dataset,batch_size=2)
    before=rng_state();modes=execution_state(model);progress=[]
    complete=evaluate_joint(model,dataset,frozen,workers=0,on_progress=progress.append)
    assert complete['metrics']==report['metrics'] and complete['split']=='dev' and complete['purpose']=='selection'
    assert equal(before,rng_state()) and equal(modes,execution_state(model)) and progress[-1]['images']==2
    checks['pixel_only_full_dev_evaluator_and_rng_modes_restored']=True
    def refuses(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Unsafe evaluation accepted')
    refuses(lambda:JointMetrics().finish(2))
    refuses(lambda:evaluate_joint(model,dataset,dict(frozen,batch_size=4),workers=0))
    refuses(lambda:evaluate_joint(model,dataset,dict(frozen,dataset_sha256='different'),workers=0))
    for split in ('train','calibration','test'):
        bad=ToyDev();bad.split=split;refuses(lambda:contract(bad))
    broken=copy.deepcopy(pred);broken['count_logits'][0,0]=float('nan')
    refuses(lambda:JointMetrics().add(broken,targets))
    checks['incomplete_wrong_split_mutated_protocol_nonfinite_refused']=True
    class Failure(ToyModel):
        def forward(self,pixels,valid_pixels):
            super().forward(pixels,valid_pixels);raise RuntimeError('Injected evaluation failure')
    fail=Failure(dataset.pred);before=rng_state();modes=execution_state(fail)
    try:evaluate_joint(fail,dataset,frozen,workers=0)
    except RuntimeError as exc:assert 'Injected' in str(exc)
    else:raise AssertionError('Failure did not stop')
    assert equal(before,rng_state()) and equal(modes,execution_state(fail))
    checks['failure_restores_rng_and_modes']=True
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,source_sha256=sha(Path(__file__).with_name('joint_evaluation.py')),
                limitation='Analytic toy predictions only; no actual DEV checkpoint was evaluated or selected. Font gallery retrieval and frozen preservation limits remain separate requirements.')
    write(Path(run)/'joint_evaluation_checks.json',result);return result
