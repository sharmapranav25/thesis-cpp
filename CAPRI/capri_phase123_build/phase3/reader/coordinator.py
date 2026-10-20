"""One finite sequential GPU comparison; exclusive lock prevents duplicate jobs."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, traceback
from common import read, write

def main(root, family):
    lock=root/'coordinator.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    def status(state,**extra):
        row=dict(state=state,pid=os.getpid(),updated_unix=time.time(),run=str(root),**extra)
        write(root/'status.json',row)
    try:
        for name in ['crnn','svtr','parseq']:
            for aux in [False,True]:
                tag=name+('_aux' if aux else '_plain');out=root/tag
                if out.exists():
                    if (out/'status.json').exists() and read(out/'status.json')['state']=='complete':continue
                    raise RuntimeError('Existing incomplete run preserved; investigate before resuming: '+str(out))
                cmd=[sys.executable,str(Path(__file__).with_name('train.py')),'--model',name,'--data',str(root/'data'),'--out',str(out),'--epochs','10']+(['--aux'] if aux else [])
                with (root/(tag+'.log')).open('w') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        child_status=read(out/'status.json') if (out/'status.json').exists() else dict(state='initializing')
                        status('training',active=tag,child_pid=child.pid,child=child_status)
                        time.sleep(15)
                    if child.returncode:raise RuntimeError(tag+' failed; inspect preserved log and failure.json')
        status('aggregating_fit_only')
        from aggregate import aggregate
        scheme=aggregate(root,family)
        status('awaiting_scheme_review' if scheme['synthetic_reading_gate']['passed'] else 'blocked_inadequate_synthetic_reading',scheme=str(root/'inferred_scheme.json'),final_dataset_generated=False)
    except BaseException as e:
        write(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));status('failed',error=repr(e));raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);a=p.parse_args();main(a.root,a.family)
