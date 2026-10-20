"""Export fresh samples from the DINO controller's retained renderer schemes."""
from setup5 import ROOT,V1,U,FAMILIES
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from run import save_rows
import controls as C
import argparse,copy
import numpy as np

def make(job):
    index,recipe,policy,seed=job
    for attempt in range(20):
        r=copy.deepcopy(recipe);r['seed']=seed+attempt*100003;rng=np.random.default_rng(r['seed'])
        r['text_spec']['text']=C.R.T.make_text(rng,len(r['text_spec']['text']),r['description']['content'])
        row=C.render((r,policy))
        if row['complete'] and row['usable']:
            row['attempts']=attempt+1;row['profile_slot']=index;return row
    raise RuntimeError('Profile could not render a valid sample after20 attempts; no silent substitution')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=FAMILIES,required=True);ap.add_argument('--count',type=int,default=64);ap.add_argument('--seed',type=int,default=53003);ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.count<1:ap.error('--count must be positive')
    dest=args.output or ROOT/'generated'/args.family;dest.mkdir(parents=True,exist_ok=False)
    archive=ROOT/args.family/'archive.json';profiles=U.read(archive);templates=U.read(V1/('run_03' if args.family=='City' else args.family+'_run_01')/'fit_recipes.json')
    rng=np.random.default_rng(args.seed);jobs=[(i%len(profiles),templates[i%len(templates)],profiles[i%len(profiles)],int(rng.integers(1,2**30))) for i in range(args.count)]
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:rows=list(pool.map(make,jobs))
    save_rows(dest,'images',rows);U.dump(dest/'manifest.json',dict(family=args.family,images=len(rows),profiles=len(profiles),seed=args.seed,archive_sha256=U.sha(archive),target_answers_used=False,vlm=False))
    print(dest)

if __name__=='__main__':main()
