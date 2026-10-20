"""Shared, case-sensitive supervised baseline runner."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from adapters import Recognizer


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def environment():
    versions = {}
    for name in ['torch','torchvision','numpy','timm','transformers','pytorch-lightning',
                 'Pillow','PyYAML','mmcv-lite','mmengine']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python':sys.version,'executable':sys.executable,'packages':versions,
            'cuda_available':torch.cuda.is_available(), 'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


class ManifestDataset(Dataset):
    def __init__(self, manifest, cfg):
        self.manifest = Path(manifest).resolve()
        self.cfg = cfg
        self.rows = []
        self.pixel_hashes = set()
        with self.manifest.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {'path','label'}.issubset(reader.fieldnames):
                raise ValueError(f'{self.manifest}: CSV requires path,label headers')
            for line, row in enumerate(reader, 2):
                label = row.get('label')
                if not label or len(label) > cfg['max_label_length'] or set(label)-set(cfg['charset']):
                    raise ValueError(f'{self.manifest}:{line}: invalid/empty/overlength label {label!r}')
                if not row.get('path'):
                    raise ValueError(f'{self.manifest}:{line}: missing image path')
                path = Path(row['path'])
                if not path.is_absolute():
                    path = self.manifest.parent / path
                path = path.resolve()
                supplied_hash = (row.get('pixel_sha256') or '').strip().lower()
                if supplied_hash:
                    if len(supplied_hash) != 64 or any(c not in '0123456789abcdef' for c in supplied_hash):
                        raise ValueError(f'{self.manifest}:{line}: invalid pixel_sha256')
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    pixel_hash = supplied_hash
                else:
                    with Image.open(path) as original:
                        im = original.convert('RGB')
                        pixel_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                if pixel_hash in self.pixel_hashes:
                    raise ValueError(f'Duplicate pixels within manifest: {path}')
                self.pixel_hashes.add(pixel_hash)
                self.rows.append((path, label, pixel_hash))
        if not self.rows:
            raise ValueError(f'Empty manifest: {manifest}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, text, _ = self.rows[idx]
        h,w = self.cfg['image_size']
        with Image.open(path) as im:
            arr = np.asarray(im.convert('RGB').resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32).copy()
        return torch.from_numpy(arr).permute(2,0,1).div(127.5).sub(1), text

    def audit(self):
        records_digest = hashlib.sha256()
        for path, label, pixel_hash in self.rows:
            records_digest.update(f'{path}\0{label}\0{pixel_hash}\n'.encode())
        result = {'manifest':str(self.manifest),
                  'manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                  'records_sha256': records_digest.hexdigest(), 'samples':len(self)}
        if len(self.rows) <= 1000:
            result['images'] = [
                {'path':str(p),'label':t,'pixel_sha256':h} for p,t,h in self.rows]
        return result


def edit_distance(a,b):
    last = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,last[j]+1,last[j-1]+(x != y)))
        last = row
    return last[-1]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = errors = chars = count = 0
    for images, labels in loader:
        preds = model.predict(images.to(device))
        for pred,text in zip(preds,labels):
            correct += pred == text
            errors += edit_distance(pred,text)
            chars += len(text)
            count += 1
    return {'exact_string_accuracy':correct/count, 'character_error_rate':errors/chars,
            'samples':count, 'correct':correct, 'edit_errors':errors, 'characters':chars}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loader(ds,cfg,shuffle=False,batch_size=None):
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(ds, batch_size=batch_size or cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['num_workers'], pin_memory=torch.cuda.is_available(),
                      generator=generator, drop_last=False)


def checked_step(model, optimizer, loss, clip):
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss; optimizer step refused')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    selected = next((p for p in model.parameters() if p.grad is not None and p.grad.abs().max()>0),None)
    if selected is None:
        raise RuntimeError('No nonzero gradients')
    before = selected.detach().clone()
    optimizer.step()
    changed = not torch.equal(before,selected.detach())
    if not changed:
        raise RuntimeError('Optimizer did not change the checked parameter')
    return float(norm),changed


def smoke(cfg, folder, device):
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    # Exercise the same image/CSV loader and native loss as a real run.
    with tempfile.TemporaryDirectory(prefix='smoke_',dir=folder/'logs') as td:
        td = Path(td)
        alphabet = cfg['charset']
        samples = [alphabet[:min(5,len(alphabet))], alphabet[-min(5,len(alphabet)):][::-1]]
        samples = [s[:cfg['max_label_length']] for s in samples]
        with (td/'fixture.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['path','label'])
            for i,text in enumerate(samples):
                im=Image.new('RGB',(128,32),(255-i*20,255,255))
                ImageDraw.Draw(im).text((8,8),text,fill=(0,0,0))
                im.save(td/f'{i}.png')
                writer.writerow([f'{i}.png',text])
        ds=ManifestDataset(td/'fixture.csv',cfg)
        images,labels=next(iter(loader(ds,cfg,batch_size=2)))
        model=Recognizer(cfg).to(device)
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        loss=model.loss(images.to(device),list(labels))
        norm,changed=checked_step(model,optimizer,loss,cfg['gradient_clip'])
        model.eval()
        preds=model.predict(images.to(device))
        assert len(preds)==2 and all(isinstance(x,str) for x in preds)
        if device.type=='cuda':
            torch.cuda.synchronize()
        report={'model':cfg['model'],'variant':cfg['variant'],'status':'passed',
                'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),
                'smoke_batch_size':2,'input_shape':list(images.shape),'loss':loss.item(),
                'gradient_norm':norm,'optimizer_update_verified':changed,
                'inference_predictions':preds,'seconds':time.perf_counter()-started,
                'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                'dataset_training_started':False,'smoke_optimizer_steps':1,
                'weights_retained':False,'pretrained_weights_loaded':False,
                'dataset_configured':bool(cfg.get('train_manifest')),
                'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                'environment':environment()}
        dump(folder/'readiness.json',report)
        print(json.dumps(report,indent=2))


def train(cfg, folder, device, name, resume_path=None):
    train_ds=ManifestDataset(cfg['train_manifest'],cfg)
    val_ds=ManifestDataset(cfg['val_manifest'],cfg)
    if train_ds.pixel_hashes & val_ds.pixel_hashes:
        raise ValueError('Training and validation contain identical image pixels; provide disjoint splits.')
    logs=folder/'logs'/name
    checkpoints=folder/'checkpoints'/name
    if logs.exists() or checkpoints.exists():
        raise FileExistsError(f'Run {name!r} exists; choose a new --run-name')
    logs.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    dump(logs/'config.json',cfg)
    dump(logs/'data_audit.json',{'train':train_ds.audit(),'validation':val_ds.audit()})
    dump(logs/'environment.json',environment())
    dump(logs/'status.json', {
        'state': 'training', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'epoch': 0, 'epochs': cfg['epochs'], 'optimizer_steps': 0})
    batches=loader(train_ds,cfg,shuffle=True)
    validation=loader(val_ds,cfg)
    resume = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path else None
    if resume and resume['config']['model'] != cfg['model']:
        raise ValueError('Resume checkpoint belongs to a different model')
    model=Recognizer(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    accum=cfg['effective_batch_size']//cfg['batch_size']
    steps_per_epoch=math.ceil(len(batches)/accum)
    total_steps=cfg['epochs']*steps_per_epoch
    warmup=max(1,int(total_steps*cfg['warmup_fraction']))
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        progress=(step-warmup)/max(1,total_steps-warmup)
        return 0.5*(1+math.cos(math.pi*min(progress,1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    best=float(resume.get('metrics',{}).get('exact_string_accuracy',-1.0)) if resume else -1.0
    global_step=int(resume.get('metrics',{}).get('optimizer_steps',0)) if resume else 0
    start_epoch=int(resume.get('epoch',0))+1 if resume else 1
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
    for epoch in range(start_epoch,cfg['epochs']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss=seen=0
        for index,(images,labels) in enumerate(batches):
            # Weight a partial accumulation group by its actual sample count.
            group_start=(index//accum)*accum*cfg['batch_size']
            group_samples=min(cfg['effective_batch_size'],len(train_ds)-group_start)
            loss=model.loss(images.to(device),list(labels))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {index}')
            (loss*len(labels)/group_samples).backward()
            total_loss+=loss.item()*len(labels)
            seen+=len(labels)
            if (index+1)%accum==0 or index+1==len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step+=1
            if (index+1)%50==0:
                print(f'epoch {epoch} batch {index+1}/{len(batches)} loss {total_loss/seen:.5f}',flush=True)
        metrics=evaluate(model,validation,device)
        record={'epoch':epoch,'optimizer_steps':global_step,'train_loss':total_loss/seen,
                'lr':optimizer.param_groups[0]['lr'],**metrics}
        with (logs/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        state={'config':cfg,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'epoch':epoch,'metrics':record}
        torch.save(state,checkpoints/'last.pt')
        if metrics['exact_string_accuracy']>best:
            best=metrics['exact_string_accuracy']
            torch.save(state,checkpoints/'best.pt')
        print(json.dumps(record),flush=True)
        dump(logs/'status.json', {
            'state': 'training', 'pid': os.getpid(), 'epoch': epoch, 'epochs': cfg['epochs'],
            'optimizer_steps': global_step, 'latest': record,
            'updated_utc': datetime.now(timezone.utc).isoformat()})
    completed = {'state':'complete','epochs':cfg['epochs'],'optimizer_steps':global_step,
                 'best_exact_accuracy':best,'completed_utc':datetime.now(timezone.utc).isoformat()}
    dump(logs/'completed.json',completed)
    dump(logs/'status.json',completed)


def main(config_path):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--train-manifest')
    parser.add_argument('--val-manifest')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--run-name',default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in ['.','..']:
        parser.error('--run-name must be a simple folder name')
    folder=config_path.parent
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    checkpoint=None
    if args.evaluate:
        checkpoint=torch.load(args.evaluate,map_location='cpu',weights_only=True)
        if checkpoint['config']['model']!=cfg['model']:
            parser.error('Checkpoint belongs to a different model folder')
        cfg=checkpoint['config']
    for key in ['train_manifest','val_manifest','epochs','batch_size']:
        value=getattr(args,key)
        if value is not None:
            cfg[key]=value
    if cfg['initialization']!='random' or cfg['precision']!='fp32':
        parser.error('This baseline supports random initialization and fp32; configuration must match')
    if not cfg['charset'] or len(set(cfg['charset']))!=len(cfg['charset']):
        parser.error('charset must be nonempty with unique characters')
    if min(cfg['epochs'],cfg['batch_size'],cfg['max_label_length'])<1:
        parser.error('epochs, batch-size and max_label_length must be positive')
    if cfg['effective_batch_size']%cfg['batch_size']:
        parser.error('batch-size must divide effective_batch_size')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; install CUDA PyTorch or use --device cpu')
    seed_everything(cfg['seed'])
    torch.set_num_threads(4)
    if args.smoke:
        smoke(cfg,folder,device)
    elif args.evaluate:
        if not cfg.get('val_manifest'):
            parser.error('Provide --val-manifest')
        ds=ManifestDataset(cfg['val_manifest'],cfg)
        model=Recognizer(cfg).to(device)
        model.load_state_dict(checkpoint['model'])
        print(json.dumps(evaluate(model,loader(ds,cfg),device),indent=2))
    else:
        if not cfg.get('train_manifest') or not cfg.get('val_manifest'):
            parser.error('Dataset not configured. Provide --train-manifest and --val-manifest, or use --smoke.')
        if args.resume and not args.resume.is_file():
            parser.error(f'Resume checkpoint not found: {args.resume}')
        train(cfg,folder,device,args.run_name,args.resume)
