"""Read-only process/resource checks plus a saved supervision receipt; never starts jobs."""
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import psutil
from .common import read,write,sha,now,ROOT


def inspect(run,save=True):
    run=Path(run);active=[];stale=[];alerts=[]
    for path in sorted(run.glob('*.lock.json')):
        lock=read(path)
        try:
            proc=psutil.Process(lock['pid'])
            matches=abs(proc.create_time()-lock['process_created'])<1
            if matches:
                active.append(dict(stage=path.name.replace('.lock.json',''),pid=proc.pid,
                                   started_at=lock['created_at'],children=len(proc.children(recursive=True)),
                                   rss_MiB=proc.memory_info().rss/2**20))
            else:stale.append(path.name)
        except psutil.NoSuchProcess:stale.append(path.name)
        except psutil.AccessDenied:alerts.append('Cannot verify process for '+path.name)
    progress_path=run/'corpus/progress.json'
    progress=read(progress_path) if progress_path.exists() else {}
    if progress.get('state')=='running':
        if not any(s['stage']=='generate-corpus' for s in active):
            alerts.append('Corpus marked running but no matching live generation process')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(progress['updated_at'])).total_seconds()
        if age>900:alerts.append('No new published corpus shard for more than 15 minutes')
    reference_path=run/'font_references_train_v1/progress.json'
    references=read(reference_path) if reference_path.exists() else {}
    if references.get('state')=='running':
        if not any(s['stage']=='generate-font-references' for s in active):
            alerts.append('Font references marked running but no matching live process')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(references['updated_at'])).total_seconds()
        if age>900:alerts.append('No new published font-reference shard for more than 15 minutes')
    elif references.get('state')=='failed':alerts.append('Font reference generation failed: '+references.get('error','unknown'))
    training=[]
    for path in sorted((run/'training').glob('*/progress.json')):
        item=read(path);training.append(dict({'phase':path.parent.name},**item))
        if item.get('state') not in ('phase_complete','failed'):
            if not any(s['stage']=='train-baseline' for s in active):alerts.append('Training progress exists but no live trainer: '+path.parent.name)
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(item['updated_at'])).total_seconds()
            if age>900:alerts.append('Training progress stale for over 15 minutes: '+path.parent.name)
        if item.get('state')=='failed':alerts.append('Training failed: '+path.parent.name)
    reserve=10*2**30 if training else (read(run/'corpus_protocol.json')['reserve_bytes'] if (run/'corpus_protocol.json').exists() else 10*2**30)
    free=shutil.disk_usage(run).free
    if free<reserve:alerts.append('Free disk below reserved checkpoint budget')
    gpu={}
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
        gpu['raw_csv']=query.stdout.strip()
    except Exception as exc:gpu['query_error']=str(exc)
    # Production trainer phases use run/training/. Infrastructure smoke checkpoints
    # deliberately live elsewhere and must never appear as actual decoder epochs.
    old=read(run/'supervision_checkpoint_cache.json') if (run/'supervision_checkpoint_cache.json').exists() else {}
    verified={};epochs=0;recoveries=0
    receipts=list((run/'training').glob('**/epoch_*/receipt.json'))+list((run/'training_recovery').glob('**/epoch_*/receipt.json'))
    for receipt_path in sorted(receipts):
        if '.writing.' in receipt_path.parent.name:continue
        epoch=read(receipt_path);checkpoint=receipt_path.parent/'checkpoint.pt'
        try:
            info=checkpoint.stat();key=str(checkpoint)
            stamp=dict(bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=epoch['sha256'])
            if old.get(key)!=stamp and sha(checkpoint)!=epoch['sha256']:
                raise ValueError('Checksum mismatch')
            if not set(('raw','ema','optimizer','scheduler','scaler','sampler_state','rng')).issubset(epoch['contains']):
                raise ValueError('Incomplete resume state')
            verified[key]=stamp
            if receipt_path.is_relative_to(run/'training_recovery'):recoveries+=1
            else:epochs+=1
        except Exception as exc:alerts.append('Checkpoint verification failed: '+str(checkpoint)+' '+str(exc))
    result=dict(checked_at=now(),active_stages=active,stale_locks=stale,alerts=alerts,
                corpus={k:v for k,v in progress.items() if k!='distributions'},font_references=references,free_GiB=free/2**30,
                available_RAM_GiB=psutil.virtual_memory().available/2**30,gpu=gpu,
                saved_decoder_training_epochs=epochs,verified_recovery_snapshots=recoveries,training=training,
                dependency='Authorized pretrained DINOv2; see training progress and readiness for current phase')
    if save:
        write(run/'supervision_checkpoint_cache.json',verified)
        write(run/'supervision.json',result)
    return result
