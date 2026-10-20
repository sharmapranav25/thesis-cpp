"""Model 1 PARSeq recipe, with only a synthetic iteration-2 data adapter and max length 8."""
import os,sys,ctypes,importlib.util,json,time,traceback,hashlib,argparse
from pathlib import Path
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
DLL_HANDLES=[]
if os.name=='nt':
    directory=Path(sys.base_prefix)/'Library/bin'
    if directory.is_dir():
        DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        dll=directory/'sqlite3.dll'
        if dll.exists():DLL_HANDLES.append(ctypes.WinDLL(str(dll)))
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset,DataLoader,get_worker_info
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,Callback

PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
MODEL1=Path('E:/pranav/model 1')
DATA=PROJECT/'runs/iter2_baseline_15_20261003_04'
sys.path.insert(0,str(PROJECT))
from feature_bank.iter2 import data as D,metrics as M
from feature_bank.iter2.train import leak_check
spec=importlib.util.spec_from_file_location('model1_parseq_train',MODEL1/'src/parseq/train.py')
m1=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m1;spec.loader.exec_module(m1)

def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def status(out,**values):
    write(out/'status.json',dict(pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values));print(json.dumps(values),flush=True)

class HybridImages(Dataset):
    def __init__(self,split,cfg,augment):
        self.split=split;self.cfg=cfg;self.augment=augment
        sets=D.open_split(DATA,split);self.counts=[len(s) for s in sets];self.ends=np.cumsum(self.counts);self._sets=None;self._aug=None
        self.tf=T.Compose([T.Resize(tuple(cfg['img_size']),antialias=True),T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    def __len__(self):return int(self.ends[-1])
    def __getstate__(self):
        state=self.__dict__.copy();state['_sets']=None;state['_aug']=None;return state
    def __getitem__(self,index):
        if self._sets is None:self._sets=D.open_split(DATA,self.split)
        shard=int(np.searchsorted(self.ends,index,side='right'));local=int(index-(self.ends[shard-1] if shard else 0));dataset=self._sets[shard]
        text=''.join(M.CHARS[int(c)] for c in dataset.ix['chars'][local,:int(dataset.ix['length'][local])])
        assert 2<=len(text)<=8 and all(c in m1.CHARSET for c in text)
        pixels=dataset.image(local)
        if self.augment:
            if self._aug is None:
                self._aug=m1._build_train_augment();worker=get_worker_info()
                self._aug.set_random_seed(int(worker.seed if worker else self.cfg['seed'])%(2**32))
            pixels=self._aug(image=np.asarray(pixels).copy())['image']
        return self.tf(Image.fromarray(pixels)),text

def loader(dataset,cfg,shuffle,workers=None):
    workers=cfg['num_workers'] if workers is None else workers
    return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=workers,pin_memory=True,
        persistent_workers=workers>0,collate_fn=m1.captcha_collate,drop_last=False)

class Progress(Callback):
    def __init__(self,out,cfg):
        self.out=out;self.cfg=cfg;self.history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    def on_fit_start(self,trainer,module):self.start=time.perf_counter()
    def on_train_epoch_start(self,trainer,module):self.epoch_start=time.perf_counter();self.seen=0;self.loss_sum=0.
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=outputs['loss'] if isinstance(outputs,dict) else outputs;n=len(batch[1]);self.seen+=n;self.loss_sum+=float(loss)*n
        if batch_idx%50==0 or batch_idx+1==trainer.num_training_batches:
            status(self.out,state='training',epoch=trainer.current_epoch+1,epochs=self.cfg['max_epochs'],images_this_epoch=self.seen,
                images_per_epoch=100000,step=trainer.global_step,loss=self.loss_sum/self.seen,epoch_seconds=round(time.perf_counter()-self.epoch_start,1))
    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        record=dict(epoch=trainer.current_epoch+1,step=trainer.global_step,train_images=self.seen,loss=self.loss_sum/max(1,self.seen),
            validation={k:float(v) for k,v in trainer.callback_metrics.items() if k.startswith('val_')},
            epoch_seconds=time.perf_counter()-self.epoch_start,elapsed_seconds=time.perf_counter()-self.start)
        self.history=[row for row in self.history if row['epoch']!=record['epoch']]+[record];write(self.out/'history.json',self.history)
        (self.out/'epochs').mkdir(exist_ok=True)
        torch.save(dict(state_dict=module.state_dict(),config=self.cfg,epoch=record['epoch'],validation=record['validation']),self.out/'epochs'/f'epoch_{record["epoch"]:02d}.pt')
        status(self.out,state='epoch_complete',**record)

@torch.inference_mode()
def evaluate(module,split,cfg):
    module.eval();pred=[];truth=[]
    for images,labels in loader(HybridImages(split,cfg,False),cfg,False,workers=0):
        logits=module(images.cuda());decoded,_=module.tokenizer.decode(logits.softmax(-1));pred.extend(decoded);truth.extend(labels)
    return dict(samples=len(truth),exact_accuracy=sum(a==b for a,b in zip(truth,pred))/len(truth),
        cer=sum(M.levenshtein(a,b) for a,b in zip(truth,pred))/sum(map(len,truth)),length_accuracy=sum(len(a)==len(b) for a,b in zip(truth,pred))/len(truth))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');a=parser.parse_args();out=a.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoints/last.ckpt').exists() and not a.resume:raise RuntimeError('Existing checkpoint: use --resume')
    cfg=yaml.safe_load((MODEL1/'configs/parseq_v2.yaml').read_text());cfg.update(max_label_length=8,data_dir=str(DATA),ckpt_dir=str(out/'checkpoints'))
    assert cfg['perm_num']==1 and not cfg['perm_mirrored'] and cfg['precision']=='32-true'
    (out/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8');pl.seed_everything(cfg['seed'],workers=True);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    train=HybridImages('train',cfg,True);selection=HybridImages('selection',cfg,False);assert len(train)==100000 and len(selection)==2000
    leaks=leak_check(D.open_split(DATA,'train'),[s for split in ('steering','selection','calibration','seen_test','generalisation') for s in D.open_split(DATA,split)])
    hybrid_hash=sha(DATA/'baseline_training/epoch_15/ema.pt');module=m1.build_parseq(cfg)
    protocol=dict(name='parseq0',recipe='Model 1 parseq_v2 + original build_parseq, augmentation and upstream optimizer/loss',
        recipe_config=str(MODEL1/'configs/parseq_v2.yaml'),recipe_config_sha256=sha(MODEL1/'configs/parseq_v2.yaml'),
        recipe_trainer_sha256=sha(MODEL1/'src/parseq/train.py'),runner_sha256=sha(__file__),
        train_images=100000,selection_images=2000,epochs=15,total_training_presentations=1500000,params=sum(p.numel() for p in module.parameters()),
        pretrained=False,real_labels_used=False,leakage=leaks,precision='32-true',configured_lr=cfg['lr'],effective_peak_lr=cfg['lr']*cfg['batch_size']/256,
        schedule='Original upstream OneCycleLR, 5% warmup, cosine annealing; timestep per optimizer update',
        adaptations=['Existing 100k train + separate 2k synthetic selection instead of random 95/5 split','Maximum label length 8 instead of 5','Keep final partial batch to use all 100k','Progress/resume/final reporting'],
        checkpoint_policy='Top 3 by synthetic val_accuracy + resumable last.ckpt + epoch weight snapshots; parseq0.pt is epoch 15',
        hybrid_checkpoint_sha256=hybrid_hash)
    write(out/'protocol.json',protocol)
    callback=Progress(out,cfg);checkpoint=ModelCheckpoint(dirpath=out/'checkpoints',filename='parseq-{epoch:02d}-{val_accuracy:.3f}',monitor='val_accuracy',mode='max',save_top_k=3,save_last=True)
    options=dict(max_epochs=1 if a.smoke else cfg['max_epochs'],accelerator='gpu',devices=1,precision=cfg['precision'],gradient_clip_val=cfg['gradient_clip_val'],
        callbacks=[callback,checkpoint],log_every_n_steps=50,default_root_dir=str(out),enable_progress_bar=False,num_sanity_val_steps=0)
    if a.smoke:options.update(limit_train_batches=3,limit_val_batches=1,log_every_n_steps=1)
    trainer=pl.Trainer(**options);workers=0 if a.smoke else cfg['num_workers'];status(out,state='starting',smoke=a.smoke,epochs=15,params=protocol['params'])
    trainer.fit(module,loader(train,cfg,True,workers),loader(selection,cfg,False,workers),ckpt_path=str(out/'checkpoints/last.ckpt') if a.resume else None)
    if a.smoke:
        status(out,state='smoke_complete',peak_vram_gb=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-callback.start,metrics=callback.history[-1]);return
    assert sum(row['train_images'] for row in callback.history)==1500000
    torch.save(dict(name='parseq0',state_dict={k:v.cpu() for k,v in module.state_dict().items()},config=cfg,epoch=15,protocol=protocol),out/'parseq0.pt')
    results={}
    for split in ('seen_test','generalisation'):
        status(out,state='final_evaluation',split=split);results[split]=evaluate(module,split,cfg)
    assert sha(DATA/'baseline_training/epoch_15/ema.pt')==hybrid_hash
    write(out/'final_report.json',dict(epoch=15,results=results,best_checkpoint=checkpoint.best_model_path,hybrid_unchanged=True,checkpoint_sha256=sha(out/'parseq0.pt')))
    status(out,state='complete',epoch=15,epochs=15,checkpoint=str(out/'parseq0.pt'),results=results)

if __name__=='__main__':
    try:main()
    except BaseException:
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.is_dir():write(out/'failure.json',dict(traceback=traceback.format_exc()));status(out,state='failed',error=str(sys.exc_info()[1]))
        raise
