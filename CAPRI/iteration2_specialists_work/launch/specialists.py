"""Iteration-2 synthetic specialists, TAG, assigned-output distillation and final review.

Run with the recorded training Python; all new artifacts remain in the chat workspace.
No real images/answers are opened until final inference, after checkpoint selection.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from feature_bank.iter2 import data as D, losses as Lo, metrics as M
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.train import EMA, to_dev, lr_at, leak_check
from feature_bank.train.specialists import cluster_groups

BASE = PROJECT / 'runs/iter2_baseline_15_20261003_04'
CHECKPOINT = BASE / 'baseline_training/epoch_15/ema.pt'
EXPECTED = 'ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32'
FONT = ['weight', 'width', 'x_height', 'slant']
KEYS = ['eff_presence', 'txt_presence', 'eff_ordinal', 'txt_ordinal', 'length',
        'font_cat', 'font_stroke', 'font_ordinal', 'font_italic']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def event(ctx, stage, **kw):
    r = dict(stage=stage, elapsed_seconds=round(time.perf_counter()-ctx.start, 1), **kw)
    dump(ctx.out/'status.json', r)
    print(json.dumps(r), flush=True)
    with open(ctx.out/'events.jsonl', 'a', encoding='utf-8') as f: f.write(json.dumps(r)+'\n')


def task_losses(out, b, eff_weight=None, txt_weight=None):
    """Each scalar has its own valid-row count. Unknown != negative; font zero valid."""
    loss, counts = {}, {}
    for key, ykey in [('eff_presence','eff_present'), ('txt_presence','txt_present')]:
        l, n = Lo.presence_loss(out[key], b[ykey])
        for i in range(len(l)): loss[f'{key}/{i}'], counts[f'{key}/{i}'] = l[i], n[i]
    for key, y in [('eff_ordinal', b['eff_level']), ('txt_ordinal', b['txt_level']),
                   ('font_ordinal', b['font'][:,2:6])]:
        valid = y >= 0 if key == 'font_ordinal' else y > 0
        weight = eff_weight if key=='eff_ordinal' else txt_weight if key=='txt_ordinal' else None
        l, n = Lo.ordinal_loss(out[key], y.long().clamp(min=0), valid, weight)
        for i in range(len(l)): loss[f'{key}/{i}'], counts[f'{key}/{i}'] = l[i], n[i]
    for key, y in [('length', b['length'].long()-2), ('font_cat', b['font'][:,0].long()),
                   ('font_stroke', b['font'][:,1].long())]:
        valid = y >= 0
        counts[key] = valid.sum()
        loss[key] = F.cross_entropy(out[key][valid].float(), y[valid]) if valid.any() else out[key].sum()*0
    valid = b['font'][:,6] >= 0
    counts['font_italic'] = valid.sum()
    loss['font_italic'] = F.binary_cross_entropy_with_logits(out['font_italic'][valid].float(),
                    b['font'][valid,6].float()) if valid.any() else out['font_italic'].sum()*0
    frames = out['ctc'].shape[1] // (b['image'].shape[-1]//8)
    # CUDA CTC backward is nondeterministic in this installed Torch build.
    # CPU CTC keeps restart/replay exact while the CNN/transformer stay on CUDA.
    loss['ctc'] = Lo.ctc_loss(out['ctc'].float().cpu(), b['chars'].cpu(),
               b['length'].cpu(), (b['cols']*frames).cpu()).to(out['ctc'].device)
    counts['ctc'] = b['length'].numel()
    # One small device-to-host copy avoids a synchronization for every head.
    count_keys=[k for k in counts if torch.is_tensor(counts[k])]
    count_values=torch.stack([counts[k] for k in count_keys]).detach().cpu().tolist()
    counts.update({k:int(v) for k,v in zip(count_keys,count_values)})
    return loss, counts


def masked_loss(loss, counts, assigned, anchors=()):
    """Mask before averaging. Excluded heads never enter the autograd objective."""
    groups = {}
    for key in dict.fromkeys(list(assigned)+list(anchors)):
        if int(counts[key]) > 0:
            family = key.split('/')[0]
            groups.setdefault(family, []).append(loss[key])
    assert groups, 'No labelled active tasks'
    averaged={k:torch.stack(v).mean() for k,v in groups.items()}
    font=[v for k,v in averaged.items() if k.startswith('font_')]
    return sum(v for k,v in averaged.items() if not k.startswith('font_'))+(torch.stack(font).mean() if font else 0.)


def output_slice(o, task):
    key, _, index = task.partition('/')
    return o[key][:,int(index)] if index else o[key]


def distill(out, targets, tasks, b):
    pieces = {}
    for task in tasks:
        key = task.split('/')[0]
        s, t = output_slice(out, task).float(), output_slice(targets, task).float()
        if key in ('length','font_cat','font_stroke'):
            v = F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction='batchmean')
        else:
            # Ordinal distillation is restricted to measured positive levels;
            # font level zero remains a labelled training value.
            if 'ordinal' in key:
                i = int(task.split('/')[1])
                y = b['font'][:,i+2] if key == 'font_ordinal' else b['eff_level' if key=='eff_ordinal' else 'txt_level'][:,i]
                valid = y>=0 if key=='font_ordinal' else y>0
                if not valid.any(): continue
                s,t=s[valid],t[valid]
            v=F.binary_cross_entropy_with_logits(s, t.sigmoid())
        pieces.setdefault(key,[]).append(v)
    averaged={k:torch.stack(v).mean() for k,v in pieces.items()}
    font=[v for k,v in averaged.items() if k.startswith('font_')]
    return (sum(v for k,v in averaged.items() if not k.startswith('font_'))+(torch.stack(font).mean() if font else 0.)) if pieces else out['length'].sum()*0


class Context:
    def __init__(self, out, smoke=False):
        self.start=time.perf_counter(); self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.smoke=smoke; self.cfg=yaml.safe_load((BASE/'training_config.yaml').read_text())
        self.tc=dict(self.cfg['train']); self.tc.update(warmup_steps=100, batch=64 if smoke else 128)
        self.device=torch.device('cuda')
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(.8)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        assert sha(CHECKPOINT)==EXPECTED, 'Baseline checkpoint changed'
        self.sets=D.open_split(self.cfg['data'],'train')
        self.fixed={s:D.open_split(self.cfg['data'],s) for s in ['steering','selection','calibration','seen_test','generalisation']}
        assert sum(map(len,self.sets))==100000
        assert {s:sum(map(len,v)) for s,v in self.fixed.items()}==dict(steering=2000,selection=2000,calibration=2000,seen_test=2000,generalisation=2500)
        self.leaks=leak_check(self.sets,[x for v in self.fixed.values() for x in v])
        self.items,self.widths=D.items_of(self.sets)
        self.eff,self.txt=self.sets[0].meta['effect_heads'],self.sets[0].meta['text_heads']
        if smoke:
            self.items,self.widths=self.items[:512],self.widths[:512]
            # Smoke evaluation uses separate training images only, never benchmark sets.
            self.smoke_eval=self.items[512:] if len(self.items)>512 else [(self.sets[0],i) for i in range(512,576)]
        self.labels={k:np.stack([s.ix[k][i] for s,i in self.items]) for k in D.LABEL_KEYS}
        self.eff_weight=Lo.level_weights(self.labels['eff_level']).to(self.device)
        self.txt_weight=Lo.level_weights(self.labels['txt_level']).to(self.device)
        self.ids=[dict(id=f'{s.name}/{i}',sha256=s.meta['sha256'][i]) for s,i in self.items]
        self.tasks={}
        for key,names in [('eff_presence',self.eff),('txt_presence',self.txt),('eff_ordinal',self.eff),('txt_ordinal',self.txt),('font_ordinal',FONT)]:
            for i,name in enumerate(names): self.tasks[f'{key}/{i}']=dict(output=key,index=i,name=name)
        for key in ['length','font_cat','font_stroke','font_italic']: self.tasks[key]=dict(output=key,name=key)
        self.pres=[k for k in self.tasks if 'ordinal' not in k or k.startswith('font_ordinal')]
        self.all=list(self.tasks)+['ctc']
        self.ord=[k for k in self.tasks if 'ordinal' in k and self.buckets(k)]
        self.draws=256 if smoke else 20000
        self.merge_draws=512 if smoke else 100000
        self.epochs=1 if smoke else 4
        self.merge_epochs=1 if smoke else 8
        self.seed=731
        self.config=dict(baseline=str(CHECKPOINT),baseline_sha256=EXPECTED,baseline_config=self.cfg,
          proposed_defaults=True,specialist_epochs=self.epochs,specialist_draws=self.draws,merge_epochs=self.merge_epochs,
          merge_draws=self.merge_draws,intensity_epochs=self.epochs,intensity_merge_epochs=1 if smoke else 4,
          batch=self.tc['batch'],pixel_budget=D.PIXEL_BUDGET,gpu_memory_fraction=.8,
          learning_rates=dict(presence=2e-4,merge=3e-4,intensity=1e-4),min_lr=1e-5,
            warmup='100 steps capped to 10% of actual planned steps',ema='min(.999,(1+t)/(10+t))',
          reproducibility='Deterministic CUDA kernels; CPU CTC loss/backward; saved CPU/CUDA/Python/NumPy RNG and exact draw batches',
          sampling='static 50% targeted / 50% shared; task and measured value balanced; no new rendering',
          anchors=['ctc','length'],loss_weights='fixed; selected-head means before group sum; KD weight 1',
          selection_rule=dict(presence='mean assigned F1/accuracy/within-one, 0.5 presence threshold',
            merge='mean presence F1, length accuracy, font category/stroke/italic, font ordinal within-one',
            intensity='mean assigned positive-level within-one',
            gates='CTC exact and length may fall at most .02; ordinal macro at most .02; at most 10% of ordinal heads fall > .10',
            ties='earliest checkpoint; original baseline is always eligible',
            teacher='per assigned head retain baseline/start when specialist is weaker on selection',
            tests='seen_test/generalisation reused benchmarks, scored only after final selection'),
          real_answers_used=False,smoke=smoke)
        dump(self.out/'configuration.json',self.config)

    def model(self,checkpoint=CHECKPOINT):
        torch.manual_seed(self.seed)
        m=Iter2Model(n_effects=len(self.eff),n_text=len(self.txt),**self.cfg['model']).to(self.device)
        m.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True),strict=True)
        return m

    def buckets(self,task):
        key,_,ix=task.partition('/');i=int(ix) if ix else None
        if key.startswith('eff_'): y=self.labels['eff_present' if key.endswith('presence') else 'eff_level'][:,i]
        elif key.startswith('txt_'):y=self.labels['txt_present' if key.endswith('presence') else 'txt_level'][:,i]
        elif key=='font_ordinal':y=self.labels['font'][:,i+2]
        elif key in ('font_cat','font_stroke','font_italic'): y=self.labels['font'][:,{'font_cat':0,'font_stroke':1,'font_italic':6}[key]]
        elif key=='length':y=self.labels['length']
        else:raise ValueError(task)
        return {int(v):np.flatnonzero(y==v) for v in np.unique(y) if v>=0 and (not key.endswith('ordinal') or key=='font_ordinal' or v>0)}


def audit(ctx):
    report={};pools={}
    for task,spec in ctx.tasks.items():
        buckets=ctx.buckets(task);counts={str(k):len(v) for k,v in buckets.items()}
        report[task]=dict(**spec,counts=counts,unknown=len(ctx.items)-sum(map(len,buckets.values())),
                          inadequate=[str(k) for k,v in buckets.items() if len(v)<20])
        if spec['output'].endswith('ordinal') and spec['output']!='font_ordinal':
            y=ctx.labels['eff_level' if spec['output']=='eff_ordinal' else 'txt_level'][:,spec['index']]
            report[task].update(absent=int((y==0).sum()),unknown=int((y<0).sum()),
                               missing_positive_levels=[i for i in range(1,11) if i not in buckets])
        for value,indices in buckets.items():pools[f'{task.replace("/","__")}__{value}']=indices.astype(np.int32)
    np.savez_compressed(ctx.out/'sampler_indexes.npz',**pools)
    dump(ctx.out/'label_audit.json',report);dump(ctx.out/'task_manifest.json',ctx.tasks)
    split_manifest={}
    for split,sets in [('train',ctx.sets)]+list(ctx.fixed.items()):
        split_manifest[split]=[dict(path=str(s.cdir),n=len(s),meta_sha256=sha(s.cdir/'meta.json'),
                           index_sha256=sha(s.cdir/'index.npz'),ordered_image_hashes_sha256=hashlib.sha256('\n'.join(s.meta['sha256']).encode()).hexdigest()) for s in sets]
    dump(ctx.out/'data_manifest.json',dict(splits=split_manifest,training=ctx.ids,no_leak=ctx.leaks,
            synthetic_only=True,new_samples=0,heldout_fonts='Original split manifests retained; no generation'))
    event(ctx,'audit_complete',train_images=len(ctx.items),presence_tasks=len(ctx.pres),ordinal_tasks=len(ctx.ord),
          missing_level_heads=sum(bool(v.get('missing_positive_levels')) for v in report.values()))


def tag_batch(ctx,model,nb,tasks,eps):
    b=to_dev(nb,ctx.device);model.eval()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    params=[p for _,p in model.branches()['stem']]
    original=[p.detach().clone() for p in params]
    # FP32 avoids BF16 quantisation hiding short lookahead changes.
    losses,counts=task_losses(model(b['image'],b['cols']),b)
    valid=[i for i,t in enumerate(tasks) if counts[t]>0 and losses[t].item()>1e-9]
    grads=[]
    for ai,i in enumerate(valid):
        g=torch.autograd.grad(losses[tasks[i]],params,retain_graph=ai<len(valid)-1,allow_unused=True)
        flat=torch.cat([(v if v is not None else torch.zeros_like(p)).detach().flatten() for v,p in zip(g,params)])
        grads.append(flat/flat.norm().clamp(min=1e-12))
    values=np.asarray([float(losses[t].detach()) for t in tasks]);del losses
    z=np.full((len(tasks),len(tasks)),np.nan);c=z.copy()
    G=torch.stack(grads);cos=(G@G.T).cpu().numpy();c[np.ix_(valid,valid)]=cos
    try:
        with torch.no_grad():
            for a,i in enumerate(valid):
                off=0
                for p,orig in zip(params,original):
                    p.copy_(orig-eps*G[a,off:off+p.numel()].reshape_as(p));off+=p.numel()
                look,nn=task_losses(model(b['image'],b['cols']),b)
                for j in valid:z[i,j]=1-float(look[tasks[j]])/values[j]
                for p,orig in zip(params,original):p.copy_(orig)
    finally:
        with torch.no_grad():
            for p,orig in zip(params,original):p.copy_(orig)
        assert all(torch.equal(p,o) for p,o in zip(params,original)), 'TAG restoration failure'
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    return z,c


def grouping(ctx,checkpoint,tasks,phase):
    directory=ctx.out/phase;directory.mkdir(exist_ok=True)
    if (directory/'groups.json').exists():return json.loads((directory/'groups.json').read_text())
    model=ctx.model(checkpoint);rng=np.random.default_rng(ctx.seed)
    # Mixture of train positives/levels; grouping sees synthetic training only.
    batches=[]
    for _ in range(2 if ctx.smoke else 4):
        indices=[]
        for task in tasks:
            bins=ctx.buckets(task)
            for ids in bins.values():indices.append(int(rng.choice(ids)))
        rng.shuffle(indices)
        # Small batches obey the same padded-pixel memory budget.
        selected=[ctx.items[i] for i in indices]
        small=D.width_batches(selected,ctx.widths[indices],32,rng,budget=D.PIXEL_BUDGET)
        batches.append(D.collate(max(small,key=len)))
    sweep=[];chosen=None
    # Validate the prescribed sweep first. If a sharp pretrained stem overshoots,
    # backtrack the measurement step; never weaken the positive-self gate.
    for eps in [.1,.03,.01,.003,.001,.0003]:
        zs=[];cs=[]
        for bi,b in enumerate(batches):
            event(ctx,'tag_measure',phase=phase,epsilon=eps,batch=bi+1,batches=len(batches))
            z,c=tag_batch(ctx,model,b,tasks,eps);zs.append(z);cs.append(c)
        zstack=np.stack(zs);n=np.isfinite(zstack).sum(0)
        z=np.nansum(zstack,0)/np.maximum(n,1);z[n==0]=np.nan
        cstack=np.stack(cs);cn=np.isfinite(cstack).sum(0);c=np.nansum(cstack,0)/np.maximum(cn,1);c[cn==0]=np.nan
        eligible=np.isfinite(np.diag(z));ratio=float((np.diag(z)[eligible]>0).mean())
        sweep.append(dict(epsilon=eps,positive_self_fraction=ratio,eligible=int(eligible.sum())))
        np.savez(directory/f'tag_{eps}.npz',affinity=z,cosine=c,counts=n)
        if ratio>=.9:
            chosen=eps;break
    dump(directory/'epsilon_sweep.json',sweep)
    if chosen is None:raise RuntimeError('No valid TAG epsilon; stop rather than cluster noise')
    if not np.isfinite(np.diag(z)).all():raise RuntimeError('Eligible task not represented in TAG; increase training batches')
    S=np.nan_to_num((z+z.T)/2,nan=0.)
    (score,k,lab),tried=cluster_groups(S,tasks,kmin=2 if ctx.smoke else 4,kmax=2 if ctx.smoke else 7,smin=2 if ctx.smoke else 4,smax=18)
    groups={f'g{i+1}':[t for t,j in zip(tasks,lab) if j==i] for i in range(int(lab.max())+1)}
    flat=[t for group in groups.values() for t in group];assert sorted(flat)==sorted(tasks)
    assert all(2<=len(v)<=18 for v in groups.values())
    dump(directory/'groups.json',groups)
    dump(directory/'report.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epsilon=chosen,
           objective=float(score),candidates=tried,groups=groups,shared_parameters='cnn stem only',
           missing_pair_policy='zero affinity only when a pair had no jointly labelled batch',ctc='preservation task; excluded',
           measured_training_batches=len(batches)))
    del model;torch.cuda.empty_cache();event(ctx,'groups_complete',phase=phase,epsilon=chosen,sizes={k:len(v) for k,v in groups.items()})
    return groups


def predict_items(ctx,model,items):
    widths=np.array([s.ix['widths'][i] for s,i in items]);model.eval()
    order={id(item):k for k,item in enumerate(items)}
    # Attach the row index rather than relying on width-batch order.
    batches=D.width_batches(list(range(len(items))),widths,ctx.tc['batch'],None,shuffle=False)
    P={};texts=['']*len(items)
    with torch.no_grad():
        for indices in batches:
            b=to_dev(D.collate([items[i] for i in indices]),ctx.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b['image'],b['cols'])
            for key in KEYS:
                value=o[key].float().cpu().numpy()
                if key not in P:P[key]=np.empty((len(items),)+value.shape[1:],np.float32)
                P[key][indices]=value
            text=M.decode(o['ctc'].float().cpu().numpy(),b['cols'].cpu().numpy()*model.ctc_frames)
            for i,t in zip(indices,text):texts[i]=t
    P['text']=texts
    L={k:np.stack([s.ix[k][i] for s,i in items]) for k in D.LABEL_KEYS}
    L['text']=[''.join(M.CHARS[c] for c in row[:n]) for row,n in zip(L['chars'],L['length'])]
    return P,L


def score(ctx,P,L,thresholds=None):
    sc=M.scores(P,L,ctx.eff,ctx.txt)
    for key,ykey,block,names in [('eff_presence','eff_present','effects',ctx.eff),('txt_presence','txt_present','text_heads',ctx.txt)]:
        for i,name in enumerate(names):
            valid=L[ykey][:,i]>=0;y=L[ykey][valid,i]==1
            threshold=(thresholds or {}).get(f'{key}/{i}',.5)
            prob=1/(1+np.exp(-np.clip(P[key][valid,i],-80,80)));p=prob>=threshold
            sc[block][name].update(f1=M.f1(p,y),recall=float((p&y).sum()/max(1,y.sum())),
               fpr=float((p&~y).sum()/max(1,(~y).sum())),positive_n=int(y.sum()),negative_n=int((~y).sum()),unknown_n=int((~valid).sum()))
    return sc


def task_score(ctx,sc,task):
    key,_,ix=task.partition('/');i=int(ix) if ix else None
    if key.startswith('eff_'):v=sc['effects'][ctx.eff[i]]
    elif key.startswith('txt_'):v=sc['text_heads'][ctx.txt[i]]
    elif key=='length':return sc['reading']['length_head']
    elif key=='font_ordinal':return sc['font'][FONT[i]+'_pm1']
    else:return sc['font'][{'font_cat':'category','font_stroke':'stroke','font_italic':'italic_f1'}[key]]
    return v.get('lvl_pm1',0.) if key.endswith('ordinal') else v['f1']


def mean_tasks(ctx,sc,tasks):return float(np.mean([task_score(ctx,sc,t) for t in tasks]))


def gate(ctx,sc,reference):
    diffs=[task_score(ctx,sc,t)-task_score(ctx,reference,t) for t in ctx.ord]
    return bool(sc['reading']['exact']>=reference['reading']['exact']-.02 and
            sc['reading']['length_head']>=reference['reading']['length_head']-.02 and
            np.mean(diffs)>=-.02 and np.mean(np.array(diffs)<-.1)<=.1)


def evaluation(ctx,model):
    if ctx.smoke:
        p,l=predict_items(ctx,model,ctx.smoke_eval);s=score(ctx,p,l)
        return dict(steering=s,selection=s)
    return {split:score(ctx,*predict_items(ctx,model,D.items_of(ctx.fixed[split])[0])) for split in ['steering','selection']}


def make_draws(ctx,tasks,count,epoch,phase,targeted):
    rng=np.random.default_rng(ctx.seed+epoch+sum(map(ord,phase)))
    if not targeted:return rng.choice(len(ctx.items),count,replace=count>len(ctx.items)).astype(np.int32)
    half=count//2;indices=list(rng.integers(len(ctx.items),size=count-half))
    plans=[]
    for t in tasks:
        bins=ctx.buckets(t)
        if not bins:continue
        plans.append((t,list(bins),bins))
    assert plans
    value_steps={t:0 for t,_,_ in plans}
    for j in range(half):
        t,values,bins=plans[j%len(plans)];v=values[value_steps[t]%len(values)];value_steps[t]+=1
        indices.append(int(rng.choice(bins[v])))
    indices=np.asarray(indices,np.int32);rng.shuffle(indices);return indices


def train_phase(ctx,path,tasks,start,reference,kind='presence',cache=None,cache_tasks=None,stop_after=None):
    directory=ctx.out/path;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'selected.json').exists():return json.loads((directory/'selected.json').read_text())
    merge=cache is not None;epochs=(1 if ctx.smoke else (4 if kind=='intensity' else 8)) if merge else ctx.epochs
    count=ctx.merge_draws if merge else ctx.draws
    lr=1e-4 if kind=='intensity' else 3e-4 if merge else 2e-4
    all_tasks=ctx.all if merge else tasks
    anchors=[] if merge else ['ctc','length']
    if kind=='intensity' and not merge:
        anchors += [t for t in ctx.pres if 'ordinal' not in t]
    # Fixed draw manifests also define the exact number of padded-pixel batches.
    plans=[]
    for ep in range(1,epochs+1):
        manifest=directory/f'draws_{ep:02d}.npy'
        if manifest.exists():indices=np.load(manifest)
        else:
            indices=make_draws(ctx,tasks,count,ep,path,not merge);np.save(manifest,indices)
        plans.append(D.width_batches(indices.tolist(),ctx.widths[indices],ctx.tc['batch'],np.random.default_rng(ctx.seed+ep)))
    total=sum(map(len,plans));tc=dict(ctx.tc,lr=lr,warmup_steps=max(1,min(100,total//10)))
    model=ctx.model(start);ema=EMA(model,.999)
    # Unassigned output modules are excluded from AdamW decay; shared modules still train.
    # Per-head shared GroupedMLP slices are restored after the optimizer step below.
    assigned_outputs={t.split('/')[0] for t in list(all_tasks)+anchors}
    if 'eff_presence' not in assigned_outputs and 'eff_ordinal' not in assigned_outputs:
        for _,p in model.branches()['effect']:p.requires_grad_(False)
    head_modules={'length':'length','ctc':'ctc','font_cat':'font_cat','font_stroke':'font_stroke','font_italic':'font_italic','font_ord':'font_ordinal'}
    for prefix,output in head_modules.items():
        if output not in assigned_outputs:
            for n,p in model.named_parameters():
                if n.startswith(prefix+'.'):p.requires_grad_(False)
    decay=[];no_decay=[]
    for n,p in model.named_parameters():
        if p.requires_grad:(no_decay if p.ndim<2 or 'bias' in n or 'emb' in n or 'pos' in n else decay).append(p)
    opt=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0)],lr=lr,betas=(.9,.95))
    # Grouped heads contain all outputs in one parameter tensor. Restore inactive
    # slices so weight decay cannot silently train unassigned predictions.
    restore=[]
    if not merge:
        active=set(list(all_tasks)+anchors)
        for name,p in model.named_parameters():
            family='eff' if name.startswith('eff_head.') else 'txt' if name.startswith('txt_head.') else None
            if family:
                mask=torch.tensor([f'{family}_presence/{i}' not in active and f'{family}_ordinal/{i}' not in active for i in range(model.n_effects if family=='eff' else model.n_text)],device=p.device)
                if p.ndim and p.shape[0]==len(mask):restore.append((p,mask,p.detach()[mask].clone()))
                if name.endswith(('w2','b2')):
                    output_mask=torch.ones_like(p,dtype=torch.bool)
                    for i in range(len(mask)):
                        if f'{family}_presence/{i}' in active:output_mask[i,...,0]=False
                        if f'{family}_ordinal/{i}' in active:output_mask[i,...,1:]=False
                    restore.append((p,output_mask,p.detach()[output_mask].clone()))
            elif name.startswith('font_ord.'):
                mask=torch.tensor([f'font_ordinal/{i}' not in active for i in range(4)],device=p.device)
                restore.append((p,mask,p.detach()[mask].clone()))
    epoch=1;cursor=0;step=0;stats={}
    last=directory/'resume.pt'
    if last.exists():
        ck=torch.load(last,map_location=ctx.device,weights_only=False)
        model.load_state_dict(ck['raw']);ema.shadow=ck['ema'];ema.t=ck['ema_step'];opt.load_state_dict(ck['optimizer'])
        epoch,cursor,step=ck['epoch'],ck['cursor'],ck['scheduler_step'];stats=ck['stats']
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        event(ctx,'phase_resumed',phase=path,epoch=epoch,cursor=cursor,step=step)
    dump(directory/'phase_config.json',dict(tasks=tasks,anchors=anchors,start=str(start),start_sha256=sha(start),epochs=epochs,
        draws_per_epoch=count,total_steps=total,warmup_steps=tc['warmup_steps'],learning_rate=lr,
        teacher_cache=str(cache) if cache else None,distilled_tasks=cache_tasks or []))
    teacher={k:np.load(Path(cache)/f'{k}.npy',mmap_mode='r') for k in KEYS} if cache else None
    def save(ep,cur):
        state=dict(raw=model.state_dict(),ema=ema.shadow,optimizer=opt.state_dict(),scheduler_step=step,ema_step=ema.t,
          epoch=ep,cursor=cur,sampler_state=dict(draw_manifest=f'draws_{min(ep,epochs):02d}.npy',seed=ctx.seed),
          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
          numpy_rng=np.random.get_state(),stats=stats)
        tmp=directory/'resume.tmp';torch.save(state,tmp);tmp.replace(last)
    progressed=0
    for ep in range(epoch,epochs+1):
        model.train();t0=time.perf_counter();batches=plans[ep-1]
        if not stats:stats=dict(loss_sum=0.,batches=0,heads={})
        for bi in range(cursor,len(batches)):
            ix=batches[bi];b=to_dev(D.collate([ctx.items[i] for i in ix]),ctx.device)
            for group in opt.param_groups:group['lr']=lr_at(step,total,tc)
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b['image'],b['cols'])
            ll,nn=task_losses(o,b,ctx.eff_weight,ctx.txt_weight);objective=masked_loss(ll,nn,all_tasks,anchors)
            kd=0.
            if teacher:
                target={k:torch.from_numpy(np.array(v[ix],copy=True)).to(ctx.device) for k,v in teacher.items()}
                kd=distill(o,target,cache_tasks,b);objective=objective+kd
            assert torch.isfinite(objective), 'Non-finite specialist loss'
            opt.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            with torch.no_grad():
                for p,mask,original in restore:p[mask]=original
            ema.update(model);step+=1;progressed+=1
            stats['loss_sum']+=float(objective.detach());stats['batches']+=1
            logged_values=torch.stack([ll[t].detach() for t in ll]).cpu().tolist()
            for t,value in zip(ll,logged_values):
                acc=stats['heads'].setdefault(t,dict(loss=0.,label_count=0))
                n=nn[t];acc['loss']+=value*n;acc['label_count']+=n
            if bi%100==0:event(ctx,'training',phase=path,epoch=ep,epochs=epochs,batch=bi,batches=len(batches),loss=float(objective.detach()),kd=float(kd),epoch_seconds=round(time.perf_counter()-t0,1))
            if (bi+1)%50==0 or (stop_after and progressed>=stop_after):save(ep,bi+1)
            if stop_after and progressed>=stop_after:
                del model,opt,ema;torch.cuda.empty_cache();return dict(interrupted=True,step=step)
        ed=directory/f'epoch_{ep:02d}';ed.mkdir(exist_ok=True)
        torch.save(model.state_dict(),ed/'raw.pt');torch.save(ema.shadow,ed/'ema.pt')
        live={k:v.detach().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema.shadow)
        sc=evaluation(ctx,model);model.load_state_dict(live);del live
        headloss={k:dict(loss=v['loss']/max(1,v['label_count']),label_count=v['label_count'],active=k in list(all_tasks)+anchors) for k,v in stats['heads'].items()}
        selection=sc['selection'];objective_score=mean_tasks(ctx,selection,ctx.ord if merge and kind=='intensity' else ctx.pres if merge else tasks)
        record=dict(epoch=ep,step=step,loss=stats['loss_sum']/stats['batches'],head_losses=headloss,scores=sc,
                    selection_score=objective_score,preservation_gate=gate(ctx,selection,reference['selection']),seconds=round(time.perf_counter()-t0,1))
        record['assigned_comparison']={t:dict(start=task_score(ctx,reference['selection'],t),
                current=task_score(ctx,selection,t),delta=task_score(ctx,selection,t)-task_score(ctx,reference['selection'],t)) for t in tasks}
        if cache:
            teacher_manifest=json.loads((Path(cache)/'manifest.json').read_text())
            record['teacher_comparison']={t:dict(teacher=max(v['baseline'],v['specialist']),student=task_score(ctx,selection,t),
                delta=task_score(ctx,selection,t)-max(v['baseline'],v['specialist'])) for t,v in teacher_manifest['comparison'].items()}
        dump(ed/'report.json',record);event(ctx,'epoch_complete',phase=path,epoch=ep,epochs=epochs,loss=record['loss'],selection_score=objective_score,
             gate=record['preservation_gate'],reading=selection['reading'],seconds=record['seconds'])
        stats={};cursor=0;save(ep+1,0)
    best=dict(checkpoint=str(start),score=mean_tasks(ctx,reference['selection'],ctx.ord if merge and kind=='intensity' else ctx.pres if merge else tasks),scores=reference,epoch=0)
    for ep in range(1,epochs+1):
        r=json.loads((directory/f'epoch_{ep:02d}'/'report.json').read_text())
        if ctx.smoke or (r['preservation_gate'] and r['selection_score']>best['score']):
            best=dict(checkpoint=str(directory/f'epoch_{ep:02d}'/'ema.pt'),score=r['selection_score'],scores=r['scores'],epoch=ep)
    best['checkpoint_sha256']=sha(best['checkpoint']);dump(directory/'selected.json',best)
    del model,opt,ema;torch.cuda.empty_cache();return best


def teachers(ctx,groups,selected,start,reference,phase):
    directory=ctx.out/phase;directory.mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():
        saved=json.loads((directory/'manifest.json').read_text())
        assert saved['training_rows']==ctx.ids
        assert all(sha(directory/f'{k}.npy')==v for k,v in saved['cache_hashes'].items())
        assert all(sha(v['checkpoint'])==v['sha256'] for v in saved['routes'].values())
        return directory,saved['tasks']
    routes={};comparison={}
    for group,tasks in groups.items():
        chosen=selected[group]
        for t in tasks:
            base=task_score(ctx,reference['selection'],t);special=task_score(ctx,chosen['scores']['selection'],t)
            checkpoint=chosen['checkpoint'] if ctx.smoke or special>base else str(start)
            routes[t]=checkpoint;comparison[t]=dict(baseline=base,specialist=special,delta=special-base,teacher_checkpoint=checkpoint)
    model=ctx.model(start)
    sample=to_dev(D.collate(ctx.items[:2]),ctx.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):shape=model(sample['image'],sample['cols'])
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='w+',dtype=np.float16,shape=(len(ctx.items),)+tuple(shape[k].shape[1:])) for k in KEYS}
    for value in arrays.values():value[:]=0
    del shape,model;torch.cuda.empty_cache()
    for checkpoint in sorted(set(routes.values())):
        assigned=[t for t,p in routes.items() if p==checkpoint];model=ctx.model(checkpoint).eval()
        batches=D.width_batches(list(range(len(ctx.items))),ctx.widths,ctx.tc['batch'],None,shuffle=False)
        done=0
        with torch.no_grad():
            for ix in batches:
                b=to_dev(D.collate([ctx.items[i] for i in ix]),ctx.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b['image'],b['cols'])
                for task in assigned:
                    key,_,index=task.partition('/')
                    if index:arrays[key][ix,int(index)]=o[key][:,int(index)].float().cpu().numpy()
                    else:arrays[key][ix]=o[key].float().cpu().numpy()
                done+=len(ix)
                if done%10000<ctx.tc['batch'] or done==len(ctx.items):event(ctx,'teacher_cache',phase=phase,checkpoint=checkpoint,completed=done,total=len(ctx.items),assigned=len(assigned))
            # Recompute a width-batched cache block and compare assigned outputs.
            ix=batches[-1];b=to_dev(D.collate([ctx.items[i] for i in ix]),ctx.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):verified=model(b['image'],b['cols'])
            cached={k:torch.from_numpy(np.array(v[ix],copy=True)).to(ctx.device) for k,v in arrays.items()}
            for task in assigned:
                assert torch.equal(output_slice(cached,task).half(),output_slice(verified,task).half()), 'Cache/image alignment failure'
        del model;torch.cuda.empty_cache()
    for value in arrays.values():value.flush()
    dump(directory/'manifest.json',dict(tasks=list(routes),routes={t:dict(checkpoint=p,sha256=sha(p)) for t,p in routes.items()},
         training_rows=ctx.ids,comparison=comparison,selection_only=True,cache_hashes={k:sha(directory/f'{k}.npy') for k in KEYS}))
    del arrays;return directory,list(routes)


def checks(ctx):
    model=ctx.model();state=torch.load(CHECKPOINT,map_location=ctx.device,weights_only=True)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items());del state
    b=to_dev(D.collate(ctx.items[:8]),ctx.device)
    o=model(b['image'],b['cols']);ll,nn=task_losses(o,b)
    objective=masked_loss(ll,nn,['eff_presence/0'])
    grads=torch.autograd.grad(objective,[o['eff_presence'],o['txt_presence'],o['ctc']],allow_unused=True,retain_graph=True)
    assert grads[0] is not None and grads[0][:,1:].abs().sum()==0 and grads[1] is None and grads[2] is None
    unknown=copy.copy(b);unknown['font']=b['font'].clone();unknown['font'][:,6]=-1
    l,n=task_losses(o,unknown);assert n['font_italic']==0 and l['font_italic']==0
    level=torch.tensor([[0,-1,1]],device=ctx.device);logits=torch.zeros((1,3,10),device=ctx.device,requires_grad=True)
    l,n=Lo.ordinal_loss(logits,level.clamp(min=0),level>=0);assert n.tolist()==[1,0,1]
    ema=EMA(model,.999);assert all(torch.equal(v,ema.shadow[k]) for k,v in model.state_dict().items())
    dump(ctx.out/'loss_checks.json',dict(baseline_exact=True,assigned_gradient_only=True,unknown_italic_mask=True,font_zero_valid=True,ema_initialization=True))
    del model,o,ll,grads;torch.cuda.empty_cache()


def calibrate(ctx,P,L):
    thresholds={}
    for key,label in [('eff_presence','eff_present'),('txt_presence','txt_present')]:
        for i in range(P[key].shape[1]):
            valid=L[label][:,i]>=0;y=L[label][valid,i]==1
            p=1/(1+np.exp(-np.clip(P[key][valid,i],-80,80)))
            best=(-1.,-1.,1.000001)
            for t in np.unique(np.r_[np.quantile(p,np.linspace(0,1,401)),1.000001]):
                pr=p>=t;fpr=(pr&~y).sum()/max(1,(~y).sum())
                if fpr<=.1:
                    value=(M.f1(pr,y),float((pr&y).sum()/max(1,y.sum())),float(t))
                    if value[:2]>best[:2]:best=value
            thresholds[f'{key}/{i}']=best[2]
    return thresholds


def final(ctx,selected,baseline,presence,intensity=None):
    directory=ctx.out/'final';directory.mkdir(exist_ok=True)
    target=directory/'hybrid.pt'
    if not target.exists():
        import shutil
        shutil.copy2(selected['checkpoint'],target)
    assert sha(target)==selected['checkpoint_sha256']
    dump(directory/'selection.json',dict(selected=selected,baseline=baseline,presence=presence,intensity=intensity,
                                       rule=ctx.config['selection_rule']))
    model=ctx.model(target)
    thresholds=calibrate(ctx,*predict_items(ctx,model,D.items_of(ctx.fixed['calibration'])[0]))
    dump(directory/'calibration.json',dict(thresholds=thresholds,split='calibration',fpr_cap=.1))
    report={}
    for split in ['seen_test','generalisation']:
        path=directory/f'{split}.json'
        if path.exists():report[split]=json.loads(path.read_text());continue
        report[split]=score(ctx,*predict_items(ctx,model,D.items_of(ctx.fixed[split])[0]),thresholds=thresholds)
        dump(path,report[split])
    comparison={}
    for t in ctx.tasks:
        base=task_score(ctx,baseline['selection'],t);end=task_score(ctx,selected['scores']['selection'],t)
        comparison[t]=dict(name=ctx.tasks[t]['name'],baseline=base,final=end,delta=end-base)
    dump(directory/'per_head_comparison.json',dict(selection=comparison,reused_benchmark_results=report,
       regressions={k:v for k,v in comparison.items() if v['delta']<0},improvements={k:v for k,v in comparison.items() if v['delta']>0}))
    del model;torch.cuda.empty_cache()
    real_inference(ctx,target,thresholds)
    assert sha(CHECKPOINT)==EXPECTED
    event(ctx,'complete',checkpoint=str(target),checkpoint_sha256=sha(target),baseline_unchanged=True)


def real_inference(ctx,checkpoint,thresholds):
    from PIL import Image
    from feature_bank.iter2.generate import CATEGORIES,STROKES
    previous=Path('C:/Users/pranav/Documents/ChatGPT/thesis/iter2_full_target_recreation_01')
    directory=ctx.out/'real_inference';directory.mkdir(exist_ok=True)
    manifest=json.loads((previous/'manifest.json').read_text());dump(directory/'manifest.json',manifest)
    if not (directory/'report.json').exists():
        # Read only identity/source fields; old predictions/answers are discarded.
        sources=[{k:r[k] for k in ('family','index','source')} for r in map(json.loads,(previous/'predictions.jsonl').read_text().splitlines())]
        model=ctx.model(checkpoint).eval();rows=[]
        for row in sources:
            with Image.open(row['source']) as image:
                rgb=image.convert('RGB');w=max(8,min(960,round(rgb.width*96/rgb.height)))
                array=np.array(rgb.resize((w,96),Image.Resampling.LANCZOS))
            cols=(w+7)//8;wp=((cols*8+D.MARGIN+7)//8)*8
            tensor=torch.from_numpy(np.pad(array,((0,0),(0,wp-w),(0,0)),mode='edge').transpose(2,0,1).copy()).to(ctx.device).float().unsqueeze(0)/255
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):o=model(tensor,torch.tensor([cols],device=ctx.device))
            result=dict(row,length=int(o['length'].argmax(-1))+2)
            for key,block,names in [('eff_presence','effects',ctx.eff),('txt_presence','text_features',ctx.txt)]:
                ordinal=o['eff_ordinal' if key=='eff_presence' else 'txt_ordinal'][0]
                result[block]={h:dict(probability=float(o[key][0,i].float().sigmoid()),present=bool(o[key][0,i].float().sigmoid()>=thresholds[f'{key}/{i}']),level=int((ordinal[i]>0).sum())) for i,h in enumerate(names)}
            font=dict(category=CATEGORIES[int(o['font_cat'].argmax(-1))],stroke=STROKES[int(o['font_stroke'].argmax(-1))],italic=bool(o['font_italic'][0]>0))
            font.update({name:int((o['font_ordinal'][0,i]>0).sum()) for i,name in enumerate(FONT)})
            result['font']=font;rows.append(result)
            if len(rows)%500==0:event(ctx,'real_inference',completed=len(rows),total=len(sources))
        from collections import Counter
        assert dict(Counter(r['family'] for r in rows))==manifest['counts'] and len(rows)==3599
        with open(directory/'predictions.jsonl','w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        dump(directory/'report.json',dict(state='complete',checkpoint_sha256=sha(checkpoint),counts=manifest['counts'],total=len(rows),real_answers_used=False,presence_threshold='per-head synthetic calibration'))
        del model;torch.cuda.empty_cache()
    from feature_bank.iter2.controller import build_profiles
    if not (ctx.out/'profiles'/'index.json').exists():build_profiles(directory/'predictions.jsonl',directory/'manifest.json',ctx.out/'profiles')
    import subprocess
    if not (ctx.out/'dataset'/'report.json').exists():
        subprocess.run([sys.executable,'-m','feature_bank.iter2.controller','generate','--profiles',str(ctx.out/'profiles'),
             '--out',str(ctx.out/'dataset'),'--count','100','--workers','12'],cwd=PROJECT,check=True)
    # An existing report can describe a failed render. Do not mark the run
    # complete until every planned sample and saved pixel hash is verified.
    preview=json.loads((ctx.out/'dataset'/'report.json').read_text())
    assert preview['stage']=='complete' and not preview['failures'], 'Preview generation is incomplete; inspect dataset/report.json'
    assert set(preview['audit'])==set(manifest['counts'])
    for family,value in preview['audit'].items():
        assert value['requested']==100 and value['generated']==100 and value['failed']==0 and value['unique_pixels']==100
    rows=list(map(json.loads,(ctx.out/'dataset'/'labels.jsonl').read_text().splitlines()))
    assert len(rows)==300 and len({(r['family'],r['index']) for r in rows})==300
    for family in manifest['counts']:
        assert {r['index'] for r in rows if r['family']==family}==set(range(100))
    for row in rows:
        assert 'error' not in row and row['renderer_usable'] and row['length']==row['requested']['length']
        with Image.open(ctx.out/'dataset'/row['image']) as image:
            assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==row['pixel_sha256']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();ctx=Context(a.out,a.smoke);audit(ctx);checks(ctx)
    baseline_path=ctx.out/'baseline_scores.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text())
    else:
        m=ctx.model();baseline=evaluation(ctx,m);dump(baseline_path,baseline);del m;torch.cuda.empty_cache()
    tasks=ctx.pres
    if ctx.smoke:tasks=['eff_presence/0','eff_presence/1','txt_presence/0','txt_presence/1']
    groups=grouping(ctx,CHECKPOINT,tasks,'presence_groups');selected={}
    for g,t in groups.items():
        if ctx.smoke and g=='g1':
            first=train_phase(ctx,'presence_'+g,t,CHECKPOINT,baseline,stop_after=1)
            if first.get('interrupted'):event(ctx,'smoke_interrupted',phase=g,step=first['step'])
        selected[g]=train_phase(ctx,'presence_'+g,t,CHECKPOINT,baseline)
    cache,ctasks=teachers(ctx,groups,selected,CHECKPOINT,baseline,'presence_teachers')
    presence=train_phase(ctx,'presence_merge',ctx.pres,CHECKPOINT,baseline,cache=cache,cache_tasks=ctasks)
    if ctx.smoke:
        # Numerical resume equivalence: replay a phase with the identical saved plan.
        import shutil
        reference=ctx.out/'presence_g1'/'epoch_01'/'raw.pt'
        shutil.copytree(ctx.out/'presence_g1',ctx.out/'resume_reference',dirs_exist_ok=True)
        for p in (ctx.out/'resume_reference').glob('epoch_*'):shutil.rmtree(p)
        for name in ['resume.pt','selected.json']:
            p=ctx.out/'resume_reference'/name
            if p.exists():p.unlink()
        torch.manual_seed(ctx.seed);torch.cuda.manual_seed_all(ctx.seed)
        train_phase(ctx,'resume_reference',groups['g1'],CHECKPOINT,baseline)
        # Exact comparison is performed with RNG-state controlled by model() at phase start.
        left=torch.load(reference,map_location='cpu',weights_only=True)
        right=torch.load(ctx.out/'resume_reference'/'epoch_01'/'raw.pt',map_location='cpu',weights_only=True)
        delta=max(float((left[k]-right[k]).abs().max()) for k in left if left[k].dtype.is_floating_point)
        assert delta==0.,f'Resume mismatch {delta}'
        ig=grouping(ctx,presence['checkpoint'],['eff_ordinal/0','eff_ordinal/1','font_ordinal/0','font_ordinal/1'],'intensity_groups')
        ispec={g:train_phase(ctx,'intensity_'+g,t,presence['checkpoint'],presence['scores'],kind='intensity') for g,t in ig.items()}
        icache,itasks=teachers(ctx,ig,ispec,presence['checkpoint'],presence['scores'],'intensity_teachers')
        train_phase(ctx,'intensity_merge',list(itasks),presence['checkpoint'],presence['scores'],kind='intensity',cache=icache,cache_tasks=itasks)
        dump(ctx.out/'smoke_report.json',dict(passed=True,resume_max_abs_difference=delta,groups=groups,
          cache_rows=len(ctx.items),merged_checkpoint=presence['checkpoint'],baseline_unchanged=sha(CHECKPOINT)==EXPECTED))
        event(ctx,'smoke_complete',resume_exact=True);return
    # Ordinal stage is needed only for adequately represented heads still below .75.
    weak=[t for t in ctx.ord if task_score(ctx,presence['scores']['selection'],t)<.75]
    chosen=presence;intensity=None
    dump(ctx.out/'intensity_decision.json',dict(threshold=.75,weak_tasks=weak,selection_only=True,
       unsupported_levels='See audit; static corpus cannot supply absent measured levels'))
    if len(weak)>=16:
        ig=grouping(ctx,presence['checkpoint'],weak,'intensity_groups');ispec={}
        for g,t in ig.items():ispec[g]=train_phase(ctx,'intensity_'+g,t,presence['checkpoint'],presence['scores'],kind='intensity')
        icache,itasks=teachers(ctx,ig,ispec,presence['checkpoint'],presence['scores'],'intensity_teachers')
        intensity=train_phase(ctx,'intensity_merge',weak,presence['checkpoint'],presence['scores'],kind='intensity',cache=icache,cache_tasks=itasks)
        # Require presence preservation when selecting an ordinal-improved final model.
        if intensity['score']>mean_tasks(ctx,presence['scores']['selection'],ctx.ord) and mean_tasks(ctx,intensity['scores']['selection'],ctx.pres)>=presence['score']-.01 and gate(ctx,intensity['scores']['selection'],baseline['selection']):chosen=intensity
    final(ctx,chosen,baseline,presence,intensity)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--out' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--out')+1])
            if output.exists():
                previous=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
                dump(output/'status.json',dict(stage='failed',error=repr(exc),previous=previous))
        raise
