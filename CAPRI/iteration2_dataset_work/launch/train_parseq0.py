"""Plain PARSeq baseline with explicit full-pass or disjoint-chunk epoch semantics."""
import argparse, concurrent.futures, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
DLL_HANDLES=[]
if os.name=='nt':
    import ctypes
    for directory in (Path(sys.base_prefix)/'Library/bin',Path(sys.base_prefix)/'DLLs'):
        if directory.is_dir():DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    sqlite_dll=Path(sys.base_prefix)/'Library/bin/sqlite3.dll'
    if sqlite_dll.exists():DLL_HANDLES.append(ctypes.WinDLL(str(sqlite_dll)))
import numpy as np
import torch
from PIL import Image

PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
UPSTREAM=PROJECT/'releases/CAPRI_phase123_v1/phase3/upstream/parseq'
DATA=PROJECT/'runs/iter2_baseline_15_20261003_04'
sys.path[:0]=[str(PROJECT),str(UPSTREAM)]
from feature_bank.iter2 import data as D, metrics as M
from feature_bank.iter2.train import leak_check
from strhub.models.utils import create_model

MODEL_ARGS=dict(charset_train=M.CHARS,charset_test=M.CHARS,max_label_length=8,img_size=[32,192],
    batch_size=128,lr=3e-4,weight_decay=.01,warmup_pct=.025,embed_dim=192,
    enc_num_heads=3,enc_depth=6,dec_num_heads=6,perm_num=6,decode_ar=True,refine_iters=1)

def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def status(out,**values):
    write(out/'status.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**values));print(json.dumps(values),flush=True)

def cache(out,split,sets):
    folder=out/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
    items,_=D.items_of(sets);n=len(items)
    if (folder/'complete.json').exists():
        complete=json.loads((folder/'complete.json').read_text());assert complete['count']==n
        return np.load(folder/'images.npy',mmap_mode='r'),json.loads((folder/'labels.json').read_text())
    pixels=np.lib.format.open_memmap(folder/'images.npy',mode='w+',dtype=np.uint8,shape=(n,32,192,3))
    def convert(item):
        dataset,i=item
        text=''.join(M.CHARS[int(c)] for c in dataset.ix['chars'][i,:int(dataset.ix['length'][i])])
        assert 2<=len(text)<=8 and all(c in M.CHARS for c in text)
        image=np.asarray(Image.fromarray(dataset.image(i)).resize((192,32),Image.Resampling.BICUBIC),np.uint8)
        return image,text
    labels=[];started=time.perf_counter()
    # Bound submissions so 100k pending futures cannot consume excessive RAM.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for begin in range(0,n,2000):
            for index,(image,text) in enumerate(pool.map(convert,items[begin:begin+2000]),begin):pixels[index]=image;labels.append(text)
            if len(labels)%10000==0 or len(labels)==n:status(out,state='preparing_cache',split=split,completed=len(labels),total=n,seconds=round(time.perf_counter()-started,1))
    pixels.flush();write(folder/'labels.json',labels);write(folder/'complete.json',dict(count=n,source_shards=[str(s.cdir) for s in sets],resize='RGB bicubic stretch to 32x192; normalize to [-1,1] at input'))
    return pixels,labels

def batch(pixels,indices):
    images=np.asarray(pixels[indices]).transpose(0,3,1,2).copy()
    return torch.from_numpy(images).cuda().float().div_(127.5).sub_(1)

@torch.inference_mode()
def evaluate(model,pixels,labels):
    model.eval();pred=[]
    for begin in range(0,len(labels),128):
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch(pixels,np.arange(begin,min(begin+128,len(labels)))))
        decoded,_=model.tokenizer.decode(logits.float().softmax(-1));pred.extend(decoded)
    return dict(samples=len(labels),exact_accuracy=sum(a==b for a,b in zip(labels,pred))/len(labels),
        cer=sum(M.levenshtein(a,b) for a,b in zip(labels,pred))/sum(map(len,labels)),
        length_accuracy=sum(len(a)==len(b) for a,b in zip(labels,pred))/len(labels))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--full-passes',action='store_true');parser.add_argument('--cache-from',type=Path);a=parser.parse_args();out=a.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Existing run; use --resume or a fresh output directory')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8);torch.backends.cuda.matmul.allow_tf32=True
    seed=731;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    training=D.open_split(DATA,'train');fixed={s:D.open_split(DATA,s) for s in ('selection','seen_test','generalisation')}
    leakage=leak_check(training,[s for sets in fixed.values() for s in sets]);assert sum(map(len,training))==100000
    model=create_model('parseq',pretrained=False,**MODEL_ARGS).cuda();model.log=lambda *args,**kw:None;model.rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    baseline_hash=sha(DATA/'baseline_training/epoch_15/ema.pt')
    allocation=[np.random.default_rng(seed+epoch).permutation(100000) for epoch in range(1,16)] if a.full_passes else np.array_split(np.random.default_rng(seed).permutation(100000),15)
    assert len(np.unique(np.concatenate(allocation)))==100000
    assert max(map(len,allocation))-min(map(len,allocation))<=1
    steps_in_epochs=[math.ceil(len(part)/128) for part in allocation];total=sum(steps_in_epochs);warmup=max(1,round(total*.025))
    np.savez(out/'epoch_allocation.npz',**{f'epoch_{i+1:02d}':part for i,part in enumerate(allocation)})
    presentations=sum(map(len,allocation))
    policy='15 full passes; every image appears once per epoch, reshuffled each epoch' if a.full_passes else 'Shuffle once; 15 disjoint equal-size chunks; every image trained on once in total'
    write(out/'epoch_allocation.json',dict(counts=list(map(len,allocation)),total_unique_images=100000,total_presentations=presentations,full_passes=a.full_passes,steps=steps_in_epochs,seed=seed))
    protocol=dict(name='parseq0',hybrid_baseline_name='model0',data=str(DATA),train_count=100000,epochs=15,seed=seed,model_args=MODEL_ARGS,
        total_training_image_presentations=presentations,epoch_image_counts=list(map(len,allocation)),epoch_policy=policy,cache_from=str(a.cache_from) if a.cache_from else None,
        params=sum(p.numel() for p in model.parameters()),pretrained=False,auxiliary_heads=False,curriculum=False,augmentation=False,
        optimizer='AdamW',lr=3e-4,weight_decay=.01,batch_size=128,warmup_steps=warmup,total_steps=total,grad_clip=5.,precision='bf16',
        input='RGB bicubic stretch 32x192, normalize [-1,1]',selection='synthetic selection CER then exact accuracy',
        final_name='parseq0.pt is epoch 15, best.pt is the selection winner',real_labels_used=False,leakage=leakage,
        hybrid_checkpoint_sha256=baseline_hash,upstream=str(UPSTREAM),runner_sha256=sha(__file__))
    write(out/'protocol.json',protocol)
    if a.smoke:
        items,_=D.items_of(training);images=np.stack([np.asarray(Image.fromarray(s.image(i)).resize((192,32),Image.Resampling.BICUBIC)) for s,i in items[:32]])
        labels=[''.join(M.CHARS[int(c)] for c in s.ix['chars'][i,:int(s.ix['length'][i])]) for s,i in items[:32]]
        model.train();started=time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.training_step((batch(images,np.arange(32)),labels),0)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        result=evaluate(model,images,labels);torch.save(dict(state_dict=model.state_dict(),model_args=MODEL_ARGS),out/'smoke.pt')
        status(out,state='smoke_complete',loss=float(loss),seconds=round(time.perf_counter()-started,1),metrics=result,params=protocol['params']);return
    status(out,state='starting',epochs=15,train_count=100000)
    cache_root=a.cache_from or out
    if a.cache_from:
        for split,sets in [('train',training)]+list(fixed.items()):
            folder=cache_root/'cache'/split;meta=json.loads((folder/'complete.json').read_text())
            assert meta['count']==sum(map(len,sets)) and meta['source_shards']==[str(s.cdir) for s in sets]
        items,_=D.items_of(training);cached=np.load(cache_root/'cache/train/images.npy',mmap_mode='r');texts=json.loads((cache_root/'cache/train/labels.json').read_text())
        for index in np.random.default_rng(seed).choice(100000,32,replace=False):
            dataset,i=items[index];text=''.join(M.CHARS[int(c)] for c in dataset.ix['chars'][i,:int(dataset.ix['length'][i])])
            assert texts[index]==text and np.array_equal(cached[index],np.asarray(Image.fromarray(dataset.image(i)).resize((192,32),Image.Resampling.BICUBIC)))
        del cached,items
        write(out/'cache_reuse_audit.json',dict(source=str(cache_root),sampled_pairs=32,pixels_and_labels_match=True))
    train_pixels,train_labels=cache(cache_root,'train',training);selection_pixels,selection_labels=cache(cache_root,'selection',fixed['selection'])
    assert len(train_labels)==100000
    step=0;first=1;best=(float('inf'),float('inf'));history=[];elapsed_previous=0.
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False);assert ck['protocol']['epoch_policy']==policy;model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        step=ck['step'];first=ck['epoch']+1;best=tuple(ck['best_key']);history=json.loads((out/'history.json').read_text());elapsed_previous=ck['elapsed_seconds']
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']]);model.rng.bit_generator.state=ck['permutation_rng']
    started=time.perf_counter()
    for epoch in range(first,16):
        model.train();order=allocation[epoch-1];steps_per_epoch=steps_in_epochs[epoch-1];loss_sum=0.;seen=0;epoch_start=time.perf_counter()
        for begin in range(0,len(order),128):
            indices=order[begin:begin+128];texts=[train_labels[int(i)] for i in indices]
            factor=(step+1)/warmup if step<warmup else .01+.99*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
            for group in optimizer.param_groups:group['lr']=3e-4*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.training_step((batch(train_pixels,indices),texts),begin//128)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch {epoch} step {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            step+=1;seen+=len(indices);loss_sum+=float(loss)*len(indices)
            if (begin//128+1)%10==0 or seen==len(order):status(out,state='training',epoch=epoch,epochs=15,images_this_epoch=seen,epoch_image_count=len(order),images_seen_total=sum(map(len,allocation[:epoch-1]))+seen,train_count=100000,step=step,steps_per_epoch=steps_per_epoch,loss=loss_sum/seen,epoch_seconds=round(time.perf_counter()-epoch_start,1))
        metrics=evaluate(model,selection_pixels,selection_labels);key=(metrics['cer'],-metrics['exact_accuracy']);improved=key<best
        if improved:best=key
        elapsed=elapsed_previous+time.perf_counter()-started;record=dict(epoch=epoch,step=step,train_images=seen,loss=loss_sum/seen,selection=metrics,epoch_seconds=time.perf_counter()-epoch_start,elapsed_seconds=elapsed)
        history.append(record);write(out/'history.json',history)
        checkpoint=dict(name='parseq0',state_dict=model.state_dict(),model_args=MODEL_ARGS,epoch=epoch,step=step,selection=metrics,protocol=protocol)
        folder=out/'checkpoints';folder.mkdir(exist_ok=True);torch.save(checkpoint,folder/f'epoch_{epoch:02d}.pt')
        if improved:torch.save(checkpoint,out/'best.pt')
        torch.save(checkpoint|dict(optimizer=optimizer.state_dict(),best_key=best,elapsed_seconds=elapsed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),permutation_rng=model.rng.bit_generator.state),out/'last.pt')
        status(out,state='epoch_complete',**record)
    assert sum(row['train_images'] for row in history)==presentations
    checkpoint=torch.load(out/'checkpoints/epoch_15.pt',map_location='cpu',weights_only=False);torch.save(checkpoint,out/'parseq0.pt')
    results={}
    for name in ('seen_test','generalisation'):
        pixels,labels=cache(cache_root,name,fixed[name]);results[name]=evaluate(model,pixels,labels)
    assert sha(DATA/'baseline_training/epoch_15/ema.pt')==baseline_hash
    write(out/'final_report.json',dict(model='parseq0',epoch=15,metrics=results,checkpoint_sha256=sha(out/'parseq0.pt'),hybrid_unchanged=True))
    status(out,state='complete',epoch=15,epochs=15,checkpoint=str(out/'parseq0.pt'),results=results,elapsed_seconds=elapsed_previous+time.perf_counter()-started)

if __name__=='__main__':
    try:main()
    except BaseException:
        if '--out' in sys.argv:
            failure_out=Path(sys.argv[sys.argv.index('--out')+1])
            if failure_out.is_dir():
                write(failure_out/'failure.json',dict(traceback=traceback.format_exc()))
                status(failure_out,state='failed',error=str(sys.exc_info()[1]))
        print(traceback.format_exc(),flush=True);raise
