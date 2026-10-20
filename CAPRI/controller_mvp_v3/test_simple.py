"""Apply the existing v1 -> v2 -> v3 protocol to the user-selected dataset."""
from boot import ROOT,V1,V2,U
import subprocess,sys
import numpy as np
from run import validate_split

def main():
    inventory=U.read(ROOT/'simple_captcha_dataset/inventory.json');rows=inventory['rows'];np.random.default_rng(117193).shuffle(rows)
    if len(rows)<640:raise ValueError('Need 640 distinct images for disjoint evaluation')
    base=dict(fit=rows[:128],selection=rows[128:192],provenance=dict(source=inventory['source'],labels_read=False,filenames='Pixel SHA256 only',archive_sha256=inventory['archive_sha256']))
    one={**base,'test':rows[192:256]};two={**base,'test':rows[256:384]};three={**two,'fresh':rows[384:640]}
    for split in [one,two,three]:validate_split(split)
    U.dump(ROOT/'SimpleCaptcha_v1_split.json',one);U.dump(ROOT/'SimpleCaptcha_v2_split.json',two);U.dump(ROOT/'SimpleCaptcha_split.json',three)
    jobs=[('v1',[str(V1/'run.py'),'--splits',str(ROOT/'SimpleCaptcha_v1_split.json'),'--out',str(V1/'SimpleCaptcha_run_01'),'--workers','6']),
        ('v2',[str(V2/'experiment.py'),'--previous',str(V1/'SimpleCaptcha_run_01'),'--splits',str(ROOT/'SimpleCaptcha_v2_split.json'),'--out',str(V2/'SimpleCaptcha_run_01'),'--workers','6']),
        ('v3',[str(ROOT/'experiment3.py'),'SimpleCaptcha','--workers','6'])]
    for name,args in jobs:
        U.dump(ROOT/'simple_pipeline_status.json',dict(stage=name))
        with (ROOT/('SimpleCaptcha_'+name+'_stdout.log')).open('w') as stdout,(ROOT/('SimpleCaptcha_'+name+'_stderr.log')).open('w') as stderr:
            result=subprocess.run([sys.executable,*args],stdout=stdout,stderr=stderr)
        if result.returncode:
            U.dump(ROOT/'simple_pipeline_status.json',dict(stage='failed',step=name,code=result.returncode));raise RuntimeError(name+' failed')
    U.dump(ROOT/'simple_pipeline_status.json',dict(stage='complete'))

if __name__=='__main__':main()

