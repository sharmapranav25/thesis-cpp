"""Model 3: curriculum-only experiment on the immutable DS2 corpus and split."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .. import runtime as U
from .ds2 import ROOT as DATA, DS1_RUN, SEEDS, bootstrap, OPENOCR

ROOT = U.WORK / 'runs/Darkbay_model3_adaptive_20261006'


def status(stage, **kw):
    U.dump(ROOT/'status.json', dict(stage=stage, updated=time.strftime('%Y-%m-%d %H:%M:%S'), **kw))
    print(stage, kw, flush=True)


def setup():
    ROOT.mkdir(exist_ok=True)
    files = ['all_lmdb/data.mdb', 'curriculum_samples.json', 'samples.json', 'charset.txt',
             'entries.json', 'inputs.json', 'length_decision.json', 'data_receipt.json',
             'bucket_summary.json', 'protocol.json', 'dataset_manifest.json',
             'synthetic_validation_shoot_policy/frozen_policy.json']
    files += [f'seed_{s}.yml' for s in SEEDS]
    files += [f'teacher_{s}{suffix}' for s in SEEDS for suffix in ['.npy', '.receipt.json']]
    hashes = {name: U.sha(DATA/name) for name in files}
    assert hashes['all_lmdb/data.mdb'] == U.read(DATA/'data_receipt.json')['lmdb_sha256']
    assert hashes['curriculum_samples.json'] == U.read(DATA/'bucket_summary.json')['curriculum_sha256']
    rows = U.read(DATA/'curriculum_samples.json')
    train = [r['index'] for r in rows if r['split'] == 'train']
    val = [r['index'] for r in rows if r['split'] == 'validation']
    assert len(set(train)) == 95000 and len(set(val)) == 5000 and not set(train) & set(val)
    protocol = dict(model='Model 3', changed_factor='curriculum only', data=str(DATA),
        input_sha256=hashes, seeds=SEEDS, epochs=20, train=95000, validation=5000,
        initial='Existing DS1 teacher ranking and DS2 easy buckets; epoch1 easy-only',
        score='FP32 unaugmented free autoregressive output; renormalize 62 character probabilities excluding special tokens; sum -log(p_true+1e-9), clamp negative roundoff to zero. Same synthetic-label NLL convention as teacher.',
        normalization='Ordinal percentiles (rank+0.5)/95000 separately over training-only teacher/student NLL; exact ties by ascending image ID',
        smoothing='Initialize from teacher training-only percentiles; D_t=0.5*D_previous+0.5*student_percentile',
        buckets='Sort smoothed difficulty then ID; array_split into 31667/31667/31666 easy/medium/hard',
        mass='risk_i=1-exp(-min(NLL_i,50)); risk_b=mean risk within current bucket. Proposal_b=0.2+0.4*(risk_b+1e-6)/(sum risks+3e-6); cap at0.5 and redistribute excess equally. Interpolate from previous mass toward proposal with max absolute change0.1. For first adaptive update use uniform previous mass; initial easy-only transition exempt.',
        bounds=dict(minimum_mass=.2, maximum_mass=.5, max_subsequent_mass_change=.1, ema_new=.5),
        draws='95000 replacement draws; deterministic rounded bucket counts, residual to largest mass, shuffle; drop last24 for94976 optimized images, batch128; same seed+epoch*100003 RNG as DS2',
        rescore='All95000 training images after each of20epochs including diagnostic epoch20. No validation or real images. Separate deterministic pass; restore all RNG and model mode; verify unchanged model and scheduler.',
        unchanged='DS2 architecture, data, exact split, augmentation, AdamW, BF16, OneCycleLR, scratch initialization, seed values,20epochs, preprocessing, decoder and synthetic checkpoint selection',
        selection='Highest synthetic whole-string accuracy; >= comparison means latest tied epoch, matching OpenOCR DS2 implementation',
        resume='Atomic latest checkpoint stores optimizer/scheduler/scaler, adaptive state and global/model/augmentation RNG together; only completed epoch boundaries; epoch archive and hash receipt. Incomplete epoch may be replayed from last committed boundary.',
        policy='Exact DS2 discovery/calibration IDs; original Laplace emitted-character reliability, confidence support, goals90/95/98/99%, minimum100, Wilson95 descriptive bound. Freeze policy and label-free real decisions before scoring.',
        limits=['Repeatedly examined historical900-image real set', 'Same5000 validation images used for checkpoint selection and shoot calibration', 'Equal epochs, additional full-pool inference compute', 'Model1 differs in data, augmentation, split and selection; Model2 is curriculum-only comparator'])
    for name, value in [('protocol.json', protocol), ('split.json', dict(train=train, validation=val))]:
        if (ROOT/name).exists():
            assert U.read(ROOT/name) == value, 'Frozen input/protocol mismatch: '+name
        else:
            U.dump(ROOT/name, value)
    source_files = [Path(__file__), Path(__file__).with_name('ds3_data.py'), Path(__file__).with_name('ds3_evaluate.py'),
                    Path(__file__).with_name('ds3_policy.py'), Path(__file__).with_name('ds2_data.py'),
                    Path(__file__).with_name('ds3_checks.py'), U.WORK/'src/capri_controller/cli.py',
                    OPENOCR/'tools/engine/trainer.py', OPENOCR/'tools/utils/ckpt.py']
    sources = {str(p): U.sha(p) for p in source_files}
    if (ROOT/'source_freeze.json').exists():
        assert U.read(ROOT/'source_freeze.json') == sources
    else:
        U.dump(ROOT/'source_freeze.json', sources)
        snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for i,path in enumerate(source_files):
            shutil.copyfile(path,snapshot/f'{i:02d}_{path.name}')
    return rows


def rng_state(trainer):
    import numpy as np
    import torch
    def aug_nodes(node):
        if hasattr(node, 'random_generator'): yield node
        for child in getattr(node, 'transforms', []):
            yield from aug_nodes(child)
    aug = list(aug_nodes(trainer.train_dataloader.dataset.aug))
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                modules={name: copy.deepcopy(m.rng.bit_generator.state) for name,m in trainer.model.named_modules()
                         if isinstance(getattr(m,'rng',None),np.random.Generator)},
                augmentation=[dict(numpy=copy.deepcopy(n.random_generator.bit_generator.state),
                                   python=n.py_random.getstate()) for n in aug])


def restore_rng(trainer, state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    for name,m in trainer.model.named_modules():
        if name in state['modules']: m.rng.bit_generator.state = state['modules'][name]
    def nodes(node):
        if hasattr(node, 'random_generator'): yield node
        for child in getattr(node,'transforms',[]): yield from nodes(child)
    for node, saved in zip(nodes(trainer.train_dataloader.dataset.aug), state['augmentation']):
        node.random_generator.bit_generator.state = saved['numpy']; node.py_random.setstate(saved['python'])


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def nll(pred, batch, decoder, chars):
    import torch
    assert torch.isfinite(pred).all()
    # Labels and positions come only from the synthetic corpus, not target-family lengths.
    length = int(batch[2].max())
    assert pred.shape[1] >= length
    character_indices = [decoder.character.index(c) for c in chars]
    q = pred[:,:length,character_indices].double()
    q = q/q.sum(-1,keepdim=True).clamp_min(1e-300)
    mapping = {decoder.character.index(c):i for i,c in enumerate(chars)}
    truth = batch[1][:,1:length+1].clone()
    mask = torch.arange(length)[None,:] < batch[2][:,None]
    indices = torch.zeros_like(truth)
    for k,v in mapping.items(): indices[truth == k] = v
    assert all(int(v) in mapping for v in truth[mask].unique())
    selected = q.gather(-1, indices.to(pred.device).unsqueeze(-1)).squeeze(-1)
    return ((-torch.log(selected+1e-9)).clamp_min(0)*mask.to(pred.device)).sum(1).cpu().numpy()


def train(seed, smoke=False, resume=False, stop_after=None):
    bootstrap()
    import lmdb
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from tools.engine.config import Config
    import tools.engine.trainer as engine
    from .ds2_data import ReaderDataset
    from .ds3_data import AdaptiveSampler, NAMES
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    rows = U.read(DATA/'curriculum_samples.json')
    frozen = U.read(ROOT/'protocol.json')
    assert U.sha(DATA/'curriculum_samples.json') == frozen['input_sha256']['curriculum_samples.json']
    output = ROOT/('smoke' if smoke else f'seed_{seed}')
    if (output/'completion.json').exists():
        record=U.read(output/'completion.json')
        assert U.sha(record['checkpoint']) == record['sha256']
        return
    if output.exists() and not resume: raise FileExistsError(str(output))
    train_rows = [r for r in rows if r['split']=='train']
    val_rows = [r for r in rows if r['split']=='validation']
    if smoke:
        train_rows = [r for b in NAMES for r in [x for x in train_rows if x['bucket']==b][:128]]
        val_rows = val_rows[:128]
    assert not {r['index'] for r in train_rows} & {r['index'] for r in val_rows}
    env = lmdb.open(str(DATA/'all_lmdb'), readonly=True, lock=False)
    charset = DATA/'charset.txt'; chars=charset.read_text().splitlines()
    cfg = yaml.safe_load((DATA/f'seed_{seed}.yml').read_text())
    baseline=copy.deepcopy(cfg)
    cfg['Global'].update(output_dir=str(output), save_res_path=str(output/'unused.txt'))
    cfg['Curriculum'] = frozen
    if smoke:
        cfg['Global']['epoch_num']=2; cfg['LRScheduler']['warmup_epoch']=.1
    else:
        check=copy.deepcopy(cfg)
        check['Global']['output_dir']=baseline['Global']['output_dir']
        check['Global']['save_res_path']=baseline['Global']['save_res_path']
        check['Curriculum']=baseline['Curriculum']
        assert check == baseline
    configfile = ROOT/('smoke.yml' if smoke else f'seed_{seed}.yml')
    configfile.write_text(yaml.safe_dump(cfg, sort_keys=False))
    times={'epoch_started':None}
    def callback(record, ids):
        epoch=record['epoch']; times['epoch_started']=time.perf_counter()
        U.dump(output/f'sampling_epoch_{epoch:02d}.json',record)
        np.save(output/f'draw_ids_epoch_{epoch:02d}.npy',ids)
        status('smoke_training' if smoke else 'training',seed=seed,epoch=epoch,mass=record['mass'])
    sampler=AdaptiveSampler(train_rows, seed, callback)
    def loader(config,mode,logger,**kw):
        chosen=train_rows if mode=='Train' else val_rows
        ds=ReaderDataset(env,[r['index'] for r in chosen],charset,seed if mode=='Train' else None)
        return DataLoader(ds,batch_size=128,sampler=sampler if mode=='Train' else None,
                          shuffle=False,num_workers=0,drop_last=mode=='Train')
    engine.build_dataloader=loader
    trainer=engine.Trainer(Config(str(configfile)), mode='train_eval', task='rec')
    assert trainer.valid_dataloader is not None
    if not resume:
        np.savez_compressed(output/'adaptive_initial.npz',ids=sampler.ids,raw=sampler.raw,
            normalized=sampler.normalized,smoothed=sampler.smooth,bucket=sampler.buckets,
            next_sampling_probability=sampler.probabilities(),next_mass=sampler.mass)
    for i,m in enumerate(trainer.model.modules()):
        if isinstance(getattr(m,'rng',None),np.random.Generator): m.rng=np.random.default_rng(seed+i)
    score_ds=ReaderDataset(env,[r['index'] for r in train_rows],charset)
    score_loader=DataLoader(score_ds,batch_size=128,shuffle=False,num_workers=0)
    history=[]; losses=[]; last={}
    protocol_sha=U.sha(ROOT/'protocol.json')
    if resume:
        ck=torch.load(output/'latest.pth',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==protocol_sha and ck['adaptive']['epoch']==ck['epoch']+1
        trainer.model.load_state_dict(ck['state_dict'],strict=True)
        trainer.optimizer.load_state_dict(ck['optimizer']); trainer.lr_scheduler.load_state_dict(ck['scheduler'])
        trainer.scaler.load_state_dict(ck['scaler'])
        sampler.load_state_dict(ck['adaptive'])
        history=ck['validation_history']; losses=ck['loss_history']
        trainer.status=dict(epoch=ck['epoch']+1,global_step=ck['global_step'],metrics=ck['metrics'])
        restore_rng(trainer,ck['rng'])
        epoch_path=output/f"epoch_{ck['epoch']}.pth"
        if not epoch_path.exists(): shutil.copyfile(output/'latest.pth',epoch_path)
        assert U.sha(epoch_path)==U.sha(output/'latest.pth')
        receipt_path=output/f"epoch_{ck['epoch']:02d}.receipt.json"
        if not receipt_path.exists():
            digest=U.sha(epoch_path)
            U.dump(receipt_path,dict(epoch=ck['epoch'],checkpoint_sha256=digest,
                scoring_checkpoint_sha256=digest,scoring_weights_sha256=ck['scoring_weights_sha256'],
                adaptive_sha256=U.sha(output/f"adaptive_epoch_{ck['epoch']:02d}.npz"),
                protocol_sha256=protocol_sha,global_step=ck['global_step'],**ck['update_receipt']))
        # An interrupted later epoch may have written a best checkpoint before committing.
        selected_epoch=max(history,key=lambda r:(r['acc'],r['epoch']))['epoch']
        selected_path=output/f'epoch_{selected_epoch}.pth'
        selected_ck=torch.load(selected_path,map_location='cpu',weights_only=False)
        selected_ck['optimizer']=None;selected_ck['scheduler']=None
        torch.save(selected_ck,output/'best.pth')
        del selected_ck
        del ck
    native_loss=trainer.loss_class; loss_sum=[0.,0]
    def record_loss(*args,**kwargs):
        result=native_loss(*args,**kwargs)
        value=float(result['loss'].detach())
        assert np.isfinite(value)
        loss_sum[0]+=value;loss_sum[1]+=1
        return result
    trainer.loss_class=record_loss
    def evaluate():
        epoch=sampler.epoch-1
        elapsed=time.perf_counter()-times['epoch_started']
        val_start=time.perf_counter();trainer.model.eval(); values=[]
        with torch.inference_mode():
            for batch in trainer.valid_dataloader:
                pred=trainer.model(batch[0].cuda())
                values.extend(nll(pred,batch,trainer.post_process_class,chars))
                raw=[t.numpy() for t in batch]
                trainer.eval_class(trainer.post_process_class(pred,raw),raw)
        metric=trainer.eval_class.get_metric()
        history.append(dict(epoch=epoch,**metric,mean_sequence_nll=float(np.mean(values)),
                            validation_seconds=time.perf_counter()-val_start))
        losses.append(dict(epoch=epoch,mean_training_loss=loss_sum[0]/loss_sum[1],
                           batches=loss_sum[1],training_seconds=elapsed))
        loss_sum[:]=[0.,0]
        U.dump(output/'validation_history.json',history);U.dump(output/'loss_history.json',losses)
        trainer.model.train()
        return metric
    trainer.eval=evaluate
    native_save=engine.save_ckpt
    def save(model,config,optimizer,scheduler,epoch,global_step,metrics,is_best=False,logger=None,prefix=None):
        if is_best:
            return native_save(model,config,optimizer,scheduler,epoch,global_step,metrics,is_best=True,logger=logger)
        if prefix is not None:
            assert prefix==f'epoch_{epoch}' and last['epoch']==epoch
            destination=output/(prefix+'.pth')
            assert not destination.exists()
            shutil.copyfile(output/'latest.pth',destination)
            digest=U.sha(destination)
            U.dump(output/f'epoch_{epoch:02d}.receipt.json',dict(epoch=epoch,checkpoint_sha256=digest,
                scoring_checkpoint_sha256=digest,scoring_weights_sha256=last['weights'],
                adaptive_sha256=U.sha(output/f'adaptive_epoch_{epoch:02d}.npz'),
                protocol_sha256=protocol_sha,global_step=global_step,**last['update']))
            if stop_after==epoch: raise InterruptedError('Intentional smoke boundary stop')
            return
        assert epoch==sampler.epoch-1 and epoch==len(history)
        saved_rng=rng_state(trainer); before=model_hash(model)
        sched=pickle.dumps(scheduler.state_dict());mode=model.training
        start=time.perf_counter();model.eval();values=[]
        status('smoke_rescoring' if smoke else 'rescoring',seed=seed,epoch=epoch,images=len(score_ds))
        try:
            with torch.inference_mode():
                for batch in score_loader:
                    values.extend(nll(model(batch[0].cuda()),batch,trainer.post_process_class,chars))
        finally:
            model.train(mode);restore_rng(trainer,saved_rng)
        assert before==model_hash(model) and sched==pickle.dumps(scheduler.state_dict())
        assert len(values)==len(train_rows)
        update=sampler.update(values)
        update['rescore_seconds']=time.perf_counter()-start
        update['mean_nll']=float(np.mean(values));update['scored_images']=len(values)
        np.savez_compressed(output/f'adaptive_epoch_{epoch:02d}.npz', ids=sampler.ids, raw=sampler.raw,
            normalized=sampler.normalized,smoothed=sampler.smooth,bucket=sampler.buckets,
            next_sampling_probability=sampler.probabilities(),next_mass=sampler.mass)
        state=dict(epoch=epoch,global_step=global_step,state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=trainer.scaler.state_dict(),
            config=config,metrics=metrics,adaptive=sampler.state_dict(),rng=saved_rng,
            validation_history=history,loss_history=losses,protocol_sha256=protocol_sha,
            scoring_weights_sha256=before,update_receipt=update)
        torch.save(state,output/'latest.pending.pth');os.replace(output/'latest.pending.pth',output/'latest.pth')
        last.update(epoch=epoch,weights=before,update=update)
    engine.save_ckpt=save
    started=time.time()
    try:
        trainer.train()
    finally:
        engine.save_ckpt=native_save;env.close()
    latest=torch.load(output/'latest.pth',map_location='cpu',weights_only=False)
    best=torch.load(output/'best.pth',map_location='cpu',weights_only=False)
    assert latest['epoch']==(2 if smoke else 20)
    selected=max(history,key=lambda r:(r['acc'],r['epoch']))
    assert best['epoch']==selected['epoch']
    for ck in [latest,best]:
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values() if v.is_floating_point())
    U.dump(output/'completion.json',dict(seed=seed,epoch=best['epoch'],checkpoint=str(output/'best.pth'),
        sha256=U.sha(output/'best.pth'),final_checkpoint=str(output/'latest.pth'),
        final_sha256=U.sha(output/'latest.pth'),final_epoch=latest['epoch'],metrics=best['metrics'],
        selection='Best synthetic accuracy; latest tie; no real labels',seconds_this_session=time.time()-started,
        training_seconds=sum(r['training_seconds'] for r in losses),
        validation_seconds=sum(r['validation_seconds'] for r in history),
        rescoring_seconds=sum(U.read(output/f'epoch_{e:02d}.receipt.json')['rescore_seconds'] for e in range(1,latest['epoch']+1))))


def child(stage,*args):
    env=os.environ.copy();env['PYTHONPATH']=str(U.WORK/'src')+os.pathsep+env.get('PYTHONPATH','')
    with (ROOT/(stage+('_'+args[-1] if args else '')+'.console.log')).open('a',encoding='utf8') as log:
        subprocess.run([sys.executable,'-m','capri_controller.training.ds3',stage,*args],cwd=U.WORK,
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','run','evaluate'])
    p.add_argument('--seed',type=int,choices=SEEDS,default=48);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int);args=p.parse_args();bootstrap()
    if args.stage=='prepare': setup();return
    if args.stage=='smoke': train(48,True,args.resume,args.stop_after);return
    if args.stage=='train': train(args.seed,False,args.resume);return
    if args.stage=='evaluate':
        from .ds3_evaluate import main as evaluate
        evaluate();return
    setup()
    if not (ROOT/'smoke/completion.json').exists(): raise RuntimeError('Complete smoke/resume checks first')
    for check in ['rule_checks.json','resume_check.json','full_pool_check.json','policy_regression.json']:
        if not (ROOT/check).exists(): raise RuntimeError('Required readiness receipt missing: '+check)
    lock=ROOT/'running.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for seed in SEEDS:
            child('train',*(['--resume'] if (ROOT/f'seed_{seed}').exists() else []),'--seed',str(seed))
        records=[U.read(ROOT/f'seed_{s}/completion.json') for s in SEEDS]
        U.dump(ROOT/'completion.json',dict(all_complete=True,seeds=records))
        child('evaluate')
        with (ROOT/'inventory.console.log').open('a') as log:
            subprocess.run([sys.executable,str(U.WORK/'capri.py'),'inventory'],cwd=U.WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',seeds=SEEDS)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
