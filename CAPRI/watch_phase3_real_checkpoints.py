"""User-authorized real inference on every observed saved checkpoint; never selects training."""
from pathlib import Path
import argparse,collections,json,os,shutil,sys,time,traceback
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(PROJECT/'style_pipeline/phase3_reader_v2'))
import numpy as np
import torch
from common import read,write,sha,groups
from models import Reader
from train import infer,calibrate,calibrated,score

def evaluate(snapshot,out,root,tag):
    checkpoint=torch.load(snapshot,map_location='cpu',weights_only=False);cfg=checkpoint['protocol'];device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Reader(cfg['model'],cfg['auxiliary'],cfg['seed']).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval();model.requires_grad_(False)
    digest=sha(snapshot);data=root/'data';rows=[json.loads(l) for l in (data/'partitions.jsonl').open()];indices=np.array([r['index'] for r in rows if r['partition']=='calibration']);cache=np.load(data/'synthetic.npy',mmap_mode='r')
    result=infer(model,cache,indices,device);labels=[rows[i]['label'] for i in indices];cal=calibrate(result,labels);cal['checkpoint_sha256']=digest;write(out/'synthetic_calibration.json',cal);write(out/'synthetic_calibration_metrics.json',score(result,labels,cal))
    real=read(data/'real_locators.json');summary=dict(reader=tag,epoch=checkpoint['epoch'],steps=checkpoint['steps'],checkpoint_sha256=digest,real_accuracy=None,role='Interim diagnostic only; not checkpoint selection or scheme update',selection_unchanged='Synthetic selection CER only',real_labels_used=False)
    for part in ['fit','validation']:
        rc=np.load(data/(part+'.npy'),mmap_mode='r');r=infer(model,rc,np.arange(len(rc)),device);cp,gp=calibrated(r,cal)
        with (out/(part+'_predictions.jsonl')).open('w') as f:
            for i,pred in enumerate(r['predictions']):
                row=dict(id=real[part][i]['id'],pixel_sha256=real[part][i]['pixel_sha256'],prediction=pred,decoded_length=len(pred),decoded_groups=groups(pred),confidence=r['confidence'][i],checkpoint_sha256=digest)
                if cp is not None:row.update(count_probabilities=cp[i].tolist(),group_probabilities=gp[i].tolist(),head_count=int(cp[i].argmax()+2))
                f.write(json.dumps(row)+'\n')
        s=dict(images=len(rc),predicted_length_histogram=dict(collections.Counter(map(len,r['predictions']))),predicted_group_combinations=dict(collections.Counter(''.join(map(str,groups(t))) for t in r['predictions'])),predicted_group_prevalence=np.mean([groups(t) for t in r['predictions']],axis=0).tolist(),mean_confidence=float(np.mean(r['confidence'])),real_accuracy=None)
        if cp is not None:s.update(head_count_histogram=dict(collections.Counter(map(int,cp.argmax(-1)+2))),decoded_head_count_disagreement=float(np.mean([len(t)!=n for t,n in zip(r['predictions'],cp.argmax(-1)+2)])))
        summary[part]=s
    assert sha(snapshot)==digest;write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',checkpoint_sha256=digest))
    del model;torch.cuda.empty_cache();return summary

def main(root):
    torch.set_num_threads(2);out=root/'interim_real_checkpoints';out.mkdir(exist_ok=True);lock=out/'watcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    summaries=read(out/'index.json') if (out/'index.json').exists() else [];done={r['checkpoint_sha256'] for r in summaries}
    def status(state,**kwargs):
        write(out/'status.json',dict(state=state,pid=os.getpid(),time=time.time(),checkpoints_evaluated=len(summaries),**kwargs));write(PROJECT/'monitor/capri_phase3_real_checkpoints.json',dict(state=state,time=time.time(),checkpoints_evaluated=len(summaries),summaries=summaries,**kwargs))
    try:
        while True:
            for name in ['crnn_plain','crnn_aux','svtr_plain','svtr_aux','parseq_plain','parseq_aux']:
                source=root/name/'best.pt'
                if not source.exists():continue
                try:
                    before=sha(source)
                    if before in done:continue
                    temp=out/(name+'_snapshot.tmp');shutil.copyfile(source,temp)
                    if sha(source)!=before or sha(temp)!=before:temp.unlink();continue
                    checkpoint=torch.load(temp,map_location='cpu',weights_only=False);epoch=checkpoint['epoch'];del checkpoint
                except (OSError,EOFError,RuntimeError):continue
                target=out/name/f'epoch_{epoch:03d}_{before[:12]}'
                if target.exists():
                    if (target/'status.json').exists() and read(target/'status.json')['state']=='complete':done.add(before);temp.unlink();continue
                    raise RuntimeError('Existing incomplete interim checkpoint evaluation preserved: '+str(target))
                target.mkdir(parents=True);snapshot=target/'frozen.pt';temp.replace(snapshot);status('evaluating',active=name,epoch=epoch)
                summary=evaluate(snapshot,target,root,name);summaries.append(summary);done.add(before);write(out/'index.json',summaries);status('watching');print(json.dumps(summary),flush=True)
            current=read(root/'status.json')
            if current['state'] not in ['training','aggregating_fit_only']:
                status('complete',training_state=current['state']);break
            time.sleep(10)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));status('failed',error=repr(e));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
