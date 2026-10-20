"""DS2 generation, frozen-teacher ranking, and three-seed curriculum run."""
import argparse
import collections
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .. import runtime as U
from .prepare_parseq import RUN as DS1_RUN, OPENOCR

NAME = 'Darkbay_ds2_noFont-intel'
ROOT = U.WORK / 'runs' / NAME
SOURCE = U.WORK / 'capri_darkbay_workflow'
SEEDS = [48,123,456]


def bootstrap():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    sys.path[:0] = [str(U.WORK/'.runtime/parseq_packages'), str(OPENOCR)]


def status(stage, **kwargs):
    U.dump(ROOT/'status.json', dict(stage=stage, updated=time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs))
    print(stage, kwargs, flush=True)


def child(stage, *extra):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(U.WORK/'src') + os.pathsep + env.get('PYTHONPATH','')
    env['NO_ALBUMENTATIONS_UPDATE'] = '1'
    with (ROOT/(stage+('_'+extra[-1] if extra else '')+'.console.log')).open('a', encoding='utf8') as log:
        subprocess.run([sys.executable, '-m', 'capri_controller.training.ds2', stage, *extra], cwd=U.WORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['entries.json','inputs.json','length_decision.json']:
        destination = ROOT/name
        if destination.exists():
            assert U.sha(destination) == U.sha(SOURCE/name)
        else:
            shutil.copyfile(SOURCE/name, destination)
    protocol = dict(dataset=NAME, images=100000, font_intelligence=False, source_dataset='Darkbay_ds1_noFont-intel',
        generation='Same frozen 300 controller recipes and allocation; fresh render seeds starting at 10000000000',
        seed_base=10000000000, entries_sha256=U.sha(ROOT/'entries.json'),
        teachers=U.read(DS1_RUN/'completion.json')['seeds'],
        scoring='Equal arithmetic mean of three per-position character distributions (EOS excluded and character distribution renormalized), then sum negative log probability of known synthetic characters. Free decoded predictions also retained. No teacher forcing.',
        buckets='NLL tertiles using <= 1/3 and <= 2/3 quantiles, matching prior Model 2; ties may cause uneven counts',
        validation='Fixed seed 62026, 5000-image synthetic holdout stratified by bucket and controller; no validation image is sampled for optimization',
        training=dict(epochs=20, seeds=SEEDS, initialization='from scratch, same architecture as DS1', batch_size=128,
                      schedule={'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.60]}, sampling='with replacement, 95000 draws per epoch; drop final partial batch',
                      augmentation='Exact Model 2 v2 operations/parameters; explicitly seeded Albumentations',
                      selection='Best synthetic validation whole-string accuracy per seed; retain final epoch20 as well',
                      optimizer='DS1 AdamW/OneCycleLR settings retained; this is a curriculum adaptation, not an exact old-architecture reproduction'),
        evaluation='After all training: same historical 900 DarkBay images, frozen predictions before labels; three-seed majority ensemble and shoot-rate sweep',
        limits=['DS2 uses the same target references and recipe bank as DS1; fresh pixels do not mean new independently fitted families',
                'Real DarkBay has already been inspected; evaluation is not a fresh held-out benchmark',
                'DS1 used all100k with final-checkpoint selection; DS2 uses95k/5k and validation selection, so results cannot isolate curriculum alone'])
    if (ROOT/'protocol.json').exists():
        assert U.read(ROOT/'protocol.json') == protocol
    else:
        U.dump(ROOT/'protocol.json', protocol)
    U.dump(ROOT/'dataset_identity.json',dict(name=NAME, images=100000, font_intelligence=False, corpus_path=str(ROOT/'corpus')))


def generate():
    os.environ['CAPRI_RUN_DIR'] = str(ROOT)
    os.environ['CAPRI_CORPUS_SEED_BASE'] = '10000000000'
    # runtime may already be imported; corpus and pipeline import this attribute.
    U.RUN_ROOT = ROOT
    from .. import corpus
    assert corpus.ROOT == ROOT and corpus.SEED_BASE == 10000000000
    sys.argv = ['generate','--workers','8']
    corpus.main()


def pack():
    import lmdb
    import numpy as np
    from PIL import Image
    import zipfile
    receipt = ROOT/'data_receipt.json'
    if receipt.exists():
        assert U.sha(ROOT/'all_lmdb/data.mdb') == U.read(receipt)['lmdb_sha256']
        return
    excluded = set()
    old_seeds = set()
    for path in sorted((SOURCE/'corpus').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for line in archive.read('labels.jsonl').decode().splitlines():
                row = json.loads(line)
                excluded.add(row['pixel_sha256']); old_seeds.add(row['seed'])
    assert len(excluded) == 100000
    db = ROOT/'all_lmdb'
    if db.exists():
        raise RuntimeError('Incomplete packing exists; inspect before recovery')
    db.mkdir()
    env = lmdb.open(str(db), map_size=8*1024**3)
    records = []; seen = set(); controllers = collections.Counter(); shards = []
    target = {r['pixel_sha256'] for r in U.read(ROOT/'inputs.json')}
    try:
        for path in sorted((ROOT/'corpus').glob('*.zip')):
            shard = U.read(path.with_suffix('.json'))
            assert U.sha(path) == shard['zip_sha256'] and shard['seed_base'] == 10000000000
            with zipfile.ZipFile(path) as archive, env.begin(write=True) as txn:
                rows = [json.loads(line) for line in archive.read('labels.jsonl').decode().splitlines()]
                for row in rows:
                    data = archive.read(row['image'])
                    im = Image.open(io.BytesIO(data)).convert('RGB')
                    digest = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                    assert digest == row['pixel_sha256'] and digest not in seen and digest not in excluded and digest not in target
                    assert row['seed'] not in old_seeds
                    assert row['text'] == row['text_spec']['text'] and row['complete'] and row['usable']
                    seen.add(digest); controllers[row['controller']] += 1
                    index = len(records)+1
                    txn.put(f'image-{index:09d}'.encode(),data)
                    txn.put(f'label-{index:09d}'.encode(),row['text'].encode())
                    records.append(dict(index=index, text=row['text'], pixel_sha256=digest, controller=row['controller'], entry_id=row['entry_id'], seed=row['seed'], shard=path.name, image=row['image'], strict=row['strict']))
            shards.append(dict(path='corpus/'+path.name, count=len(rows), sha256=shard['zip_sha256']))
        assert len(records) == 100000 and sorted(controllers.values()) == [33333,33333,33334]
        with env.begin(write=True) as txn:
            txn.put(b'num-samples',str(len(records)).encode())
        env.sync()
    finally:
        env.close()
    U.dump(ROOT/'samples.json',records)
    shutil.copyfile(DS1_RUN/'charset.txt',ROOT/'charset.txt')
    U.dump(ROOT/'dataset_manifest.json',dict(name=NAME, count=len(records), controllers=dict(controllers), shards=shards))
    U.dump(receipt,dict(images=len(records), lmdb_sha256=U.sha(db/'data.mdb'), ds1_overlap=0, target_overlap=0, unique_pixels=len(seen), samples_sha256=U.sha(ROOT/'samples.json')))


def score():
    import lmdb
    import numpy as np
    import torch
    from PIL import Image
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from .ds2_data import transform
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    samples = U.read(ROOT/'samples.json')
    assert len(samples) == 100000
    env = lmdb.open(str(ROOT/'all_lmdb'), readonly=True, lock=False)
    tf = transform()
    labels = [r['text'] for r in samples]
    length = U.read(ROOT/'length_decision.json')['default_length']
    assert all(len(t) == length for t in labels)
    for run in U.read(DS1_RUN/'completion.json')['seeds']:
        seed = run['seed']; dest = ROOT/f'teacher_{seed}'
        if dest.with_suffix('.receipt.json').exists():
            record = U.read(dest.with_suffix('.receipt.json'))
            assert U.sha(dest.with_suffix('.npy')) == record['probabilities_sha256']
            assert U.sha(dest.with_suffix('.json')) == record['predictions_sha256']
            continue
        assert U.sha(run['checkpoint']) == run['sha256']
        saved = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
        cfg = copy.deepcopy(saved['config'])
        decoder = build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = decoder.get_character_num()
        model = build_model(cfg['Architecture']); model.load_state_dict(saved['state_dict'], strict=True); model.cuda().eval()
        chars = (ROOT/'charset.txt').read_text().splitlines()
        char_indices = [decoder.character.index(c) for c in chars]
        char_map = {c:i for i,c in enumerate(chars)}
        true_probs = np.empty((len(samples),length), dtype=np.float64); predictions=[]
        with torch.inference_mode():
            for start in range(0,len(samples),128):
                chunk = samples[start:start+128]
                with env.begin() as txn:
                    x = torch.stack([tf(Image.open(io.BytesIO(txn.get(f"image-{r['index']:09d}".encode()))).convert('RGB')) for r in chunk]).cuda()
                p = model(x)
                outputs = decoder(p)
                assert p.shape[1] >= length and torch.isfinite(p).all()
                q = p[:,:length,char_indices].double()
                q = q/q.sum(-1,keepdim=True).clamp_min(1e-300)
                truth = torch.tensor([[char_map[c] for c in r['text']] for r in chunk],device='cuda')
                true_probs[start:start+len(chunk)] = q.gather(-1,truth.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                for r,(text,conf) in zip(chunk,outputs):
                    predictions.append(dict(index=r['index'], prediction=text, confidence=float(conf) if math.isfinite(conf) else None, correct=text==r['text']))
                if start%12800 == 0:
                    print('teacher',seed,start,'/100000',flush=True)
        np.save(dest.with_suffix('.npy'),true_probs)
        U.dump(dest.with_suffix('.json'),predictions)
        U.dump(dest.with_suffix('.receipt.json'),dict(checkpoint_sha256=run['sha256'],probabilities_sha256=U.sha(dest.with_suffix('.npy')),predictions_sha256=U.sha(dest.with_suffix('.json'))))
        del model,saved
        gc.collect(); torch.cuda.empty_cache()
    env.close()
    combined = sum(np.load(ROOT/f'teacher_{s}.npy') for s in SEEDS)/3
    assert np.isfinite(combined).all() and np.all((combined>=0)&(combined<=1))
    nll = -np.log(combined+1e-9).sum(1)
    t1,t2 = np.quantile(nll,[1/3,2/3])
    bucket = np.where(nll<=t1,'easy',np.where(nll<=t2,'medium','hard'))
    teachers = {s:U.read(ROOT/f'teacher_{s}.json') for s in SEEDS}
    rows=[]
    for i,r in enumerate(samples):
        candidates=[teachers[s][i] for s in SEEDS]
        assert all(c['index']==r['index'] for c in candidates)
        votes=collections.Counter(c['prediction'] for c in candidates)
        text,count=votes.most_common(1)[0]
        if count==1:text=max(candidates,key=lambda c:c['confidence'] if c['confidence'] is not None else -1)['prediction']
        rows.append(dict(index=r['index'],text=r['text'],controller=r['controller'],bucket=str(bucket[i]),nll=float(nll[i]),p_correct=float(combined[i].prod()),prediction=text,correct=text==r['text']))
    # Exact 5000 holdout, proportional within bucket/controller strata.
    strata={}
    for i,r in enumerate(rows):strata.setdefault((r['bucket'],r['controller']),[]).append(i)
    counts={k:int(len(v)*.05) for k,v in strata.items()}
    remainder=5000-sum(counts.values())
    for k in sorted(strata,key=lambda k:(-(len(strata[k])*.05-counts[k]),k))[:remainder]:counts[k]+=1
    rng=np.random.default_rng(62026); val=set()
    for k in sorted(strata):val.update(rng.choice(strata[k],counts[k],replace=False).tolist())
    assert len(val)==5000
    for i,r in enumerate(rows):r['split']='validation' if i in val else 'train'
    assert sum(r['split']=='train' for r in rows)==95000
    for name,subset in [('labels_scored',rows)]+[(b,[r for r in rows if r['bucket']==b]) for b in ['easy','medium','hard']]:
        with (ROOT/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(subset)
    U.dump(ROOT/'curriculum_samples.json',rows)
    U.dump(ROOT/'bucket_summary.json',dict(cutoffs=[float(t1),float(t2)],counts=dict(collections.Counter(bucket)),
        train=95000,validation=5000,ensemble_accuracy=100*sum(r['correct'] for r in rows)/len(rows),
        buckets={b:dict(images=sum(r['bucket']==b for r in rows),accuracy=100*np.mean([r['correct'] for r in rows if r['bucket']==b]),mean_nll=float(np.mean(nll[bucket==b]))) for b in ['easy','medium','hard']},
        curriculum_sha256=U.sha(ROOT/'curriculum_samples.json')))


def train(seed, smoke=False, preflight=False):
    import lmdb
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from tools.engine.config import Config
    import tools.engine.trainer as engine
    from .ds2_data import ReaderDataset, BucketSampler
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    data_dir=DS1_RUN/'train_lmdb' if preflight else ROOT/'all_lmdb'
    charset=DS1_RUN/'charset.txt' if preflight else ROOT/'charset.txt'
    if preflight:
        # Infrastructure-only fixture, discarded model, never production DS2 data.
        rows=[dict(index=i+1,bucket=['easy','medium','hard'][i%3],split='train' if i<384 else 'validation') for i in range(512)]
    else:
        rows=U.read(ROOT/'curriculum_samples.json')
        assert U.sha(ROOT/'curriculum_samples.json')==U.read(ROOT/'bucket_summary.json')['curriculum_sha256']
        assert U.sha(data_dir/'data.mdb')==U.read(ROOT/'data_receipt.json')['lmdb_sha256']
    output=ROOT/('preflight' if preflight else 'smoke' if smoke else f'seed_{seed}')
    if (output/'completion.json').exists():
        rec=U.read(output/'completion.json')
        assert U.sha(rec['checkpoint'])==rec['sha256']
        return
    if output.exists():raise RuntimeError('Existing incomplete seed directory; inspect before recovery: '+str(output))
    env=lmdb.open(str(data_dir),readonly=True,lock=False)
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='validation']
    if smoke:
        train_rows=[next(r for r in train_rows if r['bucket']==b) for b in ['easy','medium','hard']]*128
        val_rows=val_rows[:128]
    cfg=yaml.safe_load((DS1_RUN/f'seed_{seed}.yml').read_text())
    cfg['Global'].update(output_dir=str(output),save_res_path=str(output/'unused.txt'),eval_epoch_step=1,
        epoch_num=1 if smoke else 20,pretrained_model=None,checkpoints=None,save_epoch_step=[0,1],character_dict_path=str(charset))
    cfg['PostProcess']['character_dict_path']=str(charset)
    cfg['Eval']={'dataset':{'name':'CAPRISyntheticHoldout','data_dir_list':[str(data_dir)]}}
    cfg['Train']={'dataset':{'name':'CAPRICurriculum','data_dir_list':[str(data_dir)]},'loader':{'batch_size_per_card':128}}
    cfg['Curriculum']=U.read(ROOT/'protocol.json')['training']
    if smoke:cfg['LRScheduler']['warmup_epoch']=.1
    configfile=ROOT/('preflight.yml' if preflight else 'smoke.yml' if smoke else f'seed_{seed}.yml')
    configfile.write_text(yaml.safe_dump(cfg,sort_keys=False))
    def callback(epoch,counts):
        U.dump(output/f'sampling_epoch_{epoch:02d}.json',dict(epoch=epoch,draws=counts,with_replacement=True))
        if not smoke:status('training',seed=seed,epoch=epoch,sampled_buckets=counts)
    def loader(config,mode,logger,**kwargs):
        chosen=train_rows if mode=='Train' else val_rows
        ds=ReaderDataset(env,[r['index'] for r in chosen],charset,seed if mode=='Train' else None)
        sampler=BucketSampler([r['bucket'] for r in chosen],seed,callback=callback) if mode=='Train' else None
        return DataLoader(ds,batch_size=128,sampler=sampler,shuffle=False,num_workers=0,drop_last=mode=='Train')
    engine.build_dataloader=loader
    trainer=engine.Trainer(Config(str(configfile)),mode='train_eval',task='rec')
    assert trainer.valid_dataloader is not None
    for i,module in enumerate(trainer.model.modules()):
        if isinstance(getattr(module,'rng',None),np.random.Generator):module.rng=np.random.default_rng(seed+i)
    # Evaluate synthetic validation in FP32 without labels passed into the model.
    validation_history=[]
    def evaluate():
        trainer.model.eval()
        with torch.inference_mode():
            for batch in trainer.valid_dataloader:
                pred=trainer.model(batch[0].cuda())
                assert torch.isfinite(pred).all()
                raw=[t.numpy() for t in batch]
                trainer.eval_class(trainer.post_process_class(pred,raw),raw)
        metric=trainer.eval_class.get_metric()
        validation_history.append(dict(epoch=len(validation_history)+1,**metric))
        U.dump(output/'validation_history.json',validation_history)
        trainer.model.train()
        return metric
    trainer.eval=evaluate
    started=time.time();trainer.train()
    latest=torch.load(output/'latest.pth',map_location='cpu',weights_only=False)
    assert latest['epoch']==(1 if smoke else 20)
    assert all(torch.isfinite(v).all() for v in latest['state_dict'].values() if v.is_floating_point())
    bestpath=output/'best.pth';best=torch.load(bestpath,map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in best['state_dict'].values() if v.is_floating_point())
    U.dump(output/'completion.json',dict(seed=seed,epoch=best['epoch'],checkpoint=str(bestpath),sha256=U.sha(bestpath),
        final_checkpoint=str(output/'latest.pth'),final_sha256=U.sha(output/'latest.pth'),final_epoch=latest['epoch'],
        selection='Best synthetic holdout accuracy; never real DarkBay labels',metrics=best['metrics'],seconds=time.time()-started))
    env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','generate','pack','score','train','smoke','preflight','evaluate']);parser.add_argument('--seed',type=int,default=48)
    args=parser.parse_args();bootstrap()
    if args.stage=='preflight':setup();train(args.seed,True,True);return
    if args.stage=='generate':generate();return
    if args.stage=='pack':pack();return
    if args.stage=='score':score();return
    if args.stage in ['train','smoke']:train(args.seed,args.stage=='smoke');return
    if args.stage=='evaluate':
        from .ds2_evaluate import main as evaluate
        evaluate();return
    setup()
    lock=ROOT/'RUNNING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for stage,receipt in [('generate','corpus_audit.json'),('pack','data_receipt.json'),('score','bucket_summary.json'),('smoke','smoke/completion.json')]:
            if not (ROOT/receipt).exists():
                status(stage);child(stage)
        for seed in SEEDS:
            status('training',seed=seed);child('train','--seed',str(seed))
        completed=[U.read(ROOT/f'seed_{s}/completion.json') for s in SEEDS]
        U.dump(ROOT/'completion.json',dict(seeds=completed,epochs=20,all_complete=True))
        if not (ROOT/'evaluation_darkbay/summary.json').exists():
            status('evaluating');child('evaluate')
        status('complete',dataset=NAME,seeds=SEEDS)
    except Exception:
        status('failed',error=traceback.format_exc());raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
