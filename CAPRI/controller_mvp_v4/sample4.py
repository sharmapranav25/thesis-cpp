"""Generate fresh labeled synthetic images from a retained controller collection."""
from setup4 import ROOT,V1,U,FAMILIES,CHARSET
from corpus4 import render_job,C
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
import argparse
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--seed',type=int,default=46001)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.count<1:parser.error('--count must be positive')
    dest=args.output or ROOT/'generated'/args.family
    dest.mkdir(parents=True,exist_ok=False)
    archive_path=ROOT/'collections'/args.family/'archive.json';profiles=U.read(archive_path)
    template_path=V1/('run_03' if args.family=='City' else args.family+'_run_01')/'fit_recipes.json'
    templates=U.read(template_path);rng=np.random.default_rng(args.seed);jobs=[];used=set()
    for i in range(args.count):
        while True:
            text=''.join(rng.choice(list(CHARSET),size=int(rng.integers(2,9))))
            if text not in used:used.add(text);break
        seed=int(rng.integers(1000000,2**30));slot=i%len(profiles)
        jobs.append((i,'export',args.family,templates[i%len(templates)],text,seed,{'diverse':profiles[slot]},False,None))
    rows=[]
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
        for index,rendered,row in pool.map(render_job,jobs,chunksize=4):
            _,pixels,meta=rendered[0];name=f'{index:06d}.png';Image.fromarray(pixels).save(dest/name)
            row.update(file=name,scheme=index%len(profiles),**meta);rows.append(row)
    U.dump(dest/'labels.json',rows)
    U.dump(dest/'manifest.json',dict(family=args.family,count=len(rows),seed=args.seed,image_size=[192,48],archive_sha256=U.sha(archive_path),
        templates_sha256=U.sha(template_path),labels_sha256=U.sha(dest/'labels.json'),target_answers_used=False,vlm=False,
        note='Cycles through saved collection slots. Uses the shared full-alphabet recognition curriculum and readability repair. Exported images are synthetic, not target reconstructions.'))
    print(str(dest))

if __name__=='__main__':main()
