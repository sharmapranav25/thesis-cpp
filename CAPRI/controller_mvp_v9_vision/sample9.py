"""Export fresh labeled renders from a selected SAM or DINOv3 controller."""
from features9 import *
from build9 import render
from build7 import fresh
from run import save_rows
from concurrent.futures import ProcessPoolExecutor
import render7 as R
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['sam','dinov3'],required=True);ap.add_argument('--family',choices=FAMILIES,required=True)
    ap.add_argument('--count',type=int,default=100);ap.add_argument('--seed-offset',type=int,default=94000);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.count<=0:raise ValueError('count must be positive')
    dest=ROOT/args.backend/args.family;chosen_path=dest/'validated_selected.json'
    if not chosen_path.exists():raise RuntimeError('Run quality_gate9.py before exporting samples.')
    chosen=U.read(chosen_path);bank=U.read(dest/'recipe_bank.json')
    recipes=fresh(bank['test'],args.count,args.seed_offset)
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:rows=render(pool,recipes,chosen['controls'])
    args.output.mkdir(parents=True,exist_ok=True);save_rows(args.output,'samples',rows)
    U.dump(args.output/'manifest.json',dict(backend=args.backend,family=args.family,selected_sha256=U.sha(chosen_path),passed_validation_gate=chosen['passed_usability_gate'],count=len(rows),complete=sum(r['complete'] for r in rows),usable=sum(r['usable'] for r in rows),errors=sum('error' in r for r in rows),seed_offset=args.seed_offset))

if __name__=='__main__':main()
