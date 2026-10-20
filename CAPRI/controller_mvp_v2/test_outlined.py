"""Run the frozen MVP 1 -> MVP 2 protocol on another dataset; no family rules."""
from bootstrap import ROOT,V1,U
import hashlib,sys,subprocess,json
import numpy as np
from PIL import Image
from run import validate_split

def main():
    directory=U.PROJECT/'data/external/outlined_text_captcha/images'
    prior=U.read(U.WORK/'length_instances_02/real_inference.json')['outlined_text_captcha']['paths']
    excluded={hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest() for p in prior}
    unique={}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.webp']:continue
        pixels=np.asarray(Image.open(path).convert('RGB'));digest=hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest not in excluded:unique.setdefault(digest,dict(source=str(path),pixel_sha256=digest))
    rows=list(unique.values());np.random.default_rng(91573).shuffle(rows)
    if len(rows)<384:raise ValueError('Insufficient disjoint images for unchanged protocol')
    shared=dict(fit=rows[:128],selection=rows[128:192],provenance=dict(dataset='Outlined text',unique_available_after_exclusion=len(rows),
        excluded_known_prior_localization_images=len(excluded),prior_project_exposure='Earlier project experiments used this dataset; this is a new controller test, not a claim of globally untouched data.',target_labels_read=False))
    one={**shared,'test':rows[192:256]};two={**shared,'test':rows[256:384]}
    validate_split(one);validate_split(two)
    U.dump(ROOT/'Outlined_v1_split.json',one);U.dump(ROOT/'Outlined_split.json',two)
    U.dump(ROOT/'Outlined_pipeline.json',dict(stage='mvp1',target_counts=dict(fit=128,selection=64,mvp1_test=64,mvp2_test=128)))
    jobs=[('mvp1',[str(V1/'run.py'),'--splits',str(ROOT/'Outlined_v1_split.json'),'--out',str(V1/'Outlined_run_01'),'--workers','8']),
          ('mvp2',[str(ROOT/'experiment.py'),'--previous',str(V1/'Outlined_run_01'),'--splits',str(ROOT/'Outlined_split.json'),'--out',str(ROOT/'Outlined_run_01'),'--workers','8'])]
    for name,args in jobs:
        U.dump(ROOT/'Outlined_pipeline.json',dict(stage=name))
        with (ROOT/f'Outlined_{name}_stdout.log').open('w') as stdout,(ROOT/f'Outlined_{name}_stderr.log').open('w') as stderr:
            result=subprocess.run([sys.executable,*args],stdout=stdout,stderr=stderr)
        if result.returncode:
            U.dump(ROOT/'Outlined_pipeline.json',dict(stage='failed',step=name,exit_code=result.returncode))
            raise RuntimeError(name+' failed; see stderr log')
    U.dump(ROOT/'Outlined_pipeline.json',dict(stage='complete'))
    print('Both outlined-text tests complete; separate test splits, unchanged algorithms.')
if __name__=='__main__':main()
