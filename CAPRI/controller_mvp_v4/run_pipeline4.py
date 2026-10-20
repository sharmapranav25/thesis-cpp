from setup4 import ROOT,U
import subprocess,sys,time

def main():
    # Wait on the collection builder started independently in this session.
    while True:
        path=ROOT/'collection_status.json'
        if path.exists():
            state=U.read(path)
            if state.get('phase')=='complete' and 'family' not in state:break
        if (ROOT/'collections_stderr.log').exists() and (ROOT/'collections_stderr.log').stat().st_size:raise RuntimeError('Collection build failed; see stderr')
        time.sleep(5)
    for stage,script in [('corpus','corpus4.py'),('training','train4.py'),('inference','infer4.py')]:
        U.dump(ROOT/'pipeline_status.json',dict(stage=stage))
        with (ROOT/(stage+'_stdout.log')).open('w') as stdout,(ROOT/(stage+'_stderr.log')).open('w') as stderr:
            result=subprocess.run([sys.executable,str(ROOT/script)],stdout=stdout,stderr=stderr)
        if result.returncode:
            U.dump(ROOT/'pipeline_status.json',dict(stage='failed',step=stage,code=result.returncode));raise RuntimeError(stage+' failed')
    U.dump(ROOT/'pipeline_status.json',dict(stage='ready_for_answer_evaluation'))

if __name__=='__main__':main()
