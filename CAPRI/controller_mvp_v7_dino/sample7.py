"""Export fresh labeled synthetic samples from frozen eligible DINO v2 profiles."""
from setup7 import *
from build7 import fresh
from run import save_rows
import render7 as R
import numpy as np
import argparse,copy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=FAMILIES,required=True);ap.add_argument('--count',type=int,default=32)
    ap.add_argument('--seed',type=int,default=78001);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.count<1:raise ValueError('Count must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    dest=ROOT/args.family;archive=U.read(dest/'archive.json');banks=U.read(dest/'recipe_banks.json')
    if not archive or not all(p['eligible'] for p in archive):raise RuntimeError('No eligible frozen profiles')
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');rng=np.random.default_rng(args.seed);R.install();rows=[];attempts=0
    for i in range(args.count):
        profile=archive[i%len(archive)];source=banks[profile['bank']]['test'];size=source[0]['size']
        for _ in range(20):
            seed=int(rng.integers(1,2**31-1));attempts+=1
            if profile['bank']=='random':recipe=random_recipes(seed,1,size,fonts)[0]
            else:recipe=copy.deepcopy(source[int(rng.integers(len(source)))])
            recipe['seed']=seed;ts=recipe['text_spec']
            ts['text']=R.C.R.T.make_text(np.random.default_rng(seed+101),len(ts['text']),recipe['description']['content'])
            row=R.safe_render((recipe,profile['controls']))
            if row['complete'] and row['usable']:
                row['profile_index']=i%len(archive);row['starting_bank']=profile['bank'];rows.append(row);break
        else:raise RuntimeError('Could not render a usable labeled sample after 20 attempts; no export written')
    args.output.mkdir(parents=True);save_rows(args.output,'images',rows)
    U.dump(args.output/'manifest.json',dict(family=args.family,seed=args.seed,count=len(rows),attempts=attempts,rejected=attempts-len(rows),
        archive_sha256=U.sha(dest/'archive.json'),note='Fresh strings/seeds. Export retries incomplete/unusable renders; diagnostic evaluation never retries. Frames included in glyph masks.'))
    print('Exported',len(rows),'samples;',attempts-len(rows),'rejected attempts')

if __name__=='__main__':main()
