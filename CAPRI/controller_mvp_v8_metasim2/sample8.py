"""Generate labeled synthetic data from a frozen learned grammar policy."""
from core8 import *
from concurrent.futures import ProcessPoolExecutor
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=FAMILIES);ap.add_argument('--count',type=int,default=100);ap.add_argument('--seed',type=int,default=91001);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.count<1:raise ValueError('count must be positive')
    g=U.read(ROOT/'grammar.json');policy=Policy(g);policy.load_state_dict(torch.load(ROOT/args.family/'selected.pt',weights_only=True));policy.eval()
    colors=U.read(ROOT/args.family/'colors.json');refs=U.read(ROOT/args.family/'target_splits.json')['fit']
    # Only size is read here; no target annotation or string used.
    size=np.median([Image.open(r['source']).size for r in refs],axis=0).astype(int).tolist()
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');torch.manual_seed(args.seed)
    with torch.no_grad():actions,_,_=policy(args.count)
    recipes=random_recipes(args.seed,args.count,size,fonts,400000)
    controls=[decode(g,a,colors) for a in actions.numpy()]
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:rows=render_batch(pool,recipes,controls)
    args.output.mkdir(parents=True,exist_ok=True);save_rows(args.output,'samples',rows)
    U.dump(args.output/'manifest.json',dict(family=args.family,count=args.count,seed=args.seed,policy_sha256=U.sha(ROOT/args.family/'selected.pt'),
        complete=sum(r['complete'] for r in rows),usable=sum(r['usable'] for r in rows),errors=sum('error' in r for r in rows),
        note='All draws saved, including any unusable outputs. Check labels before solver training.'))
    print(args.output)

if __name__=='__main__':main()
