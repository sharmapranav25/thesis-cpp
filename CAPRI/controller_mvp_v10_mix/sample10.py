"""Export fresh labeled samples from the mixed controller's frozen settings."""
from mix10 import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=FAMILIES);ap.add_argument('--count',type=int,default=100);ap.add_argument('--offset',type=int,default=104000);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.count<1:raise ValueError('count must be positive')
    dest=ROOT/args.family;chosen=U.read(dest/'selected.json');p=chosen['winners']['mixed']['controls'];bank=U.read(dest/'recipes.json')
    if not chosen['passed']:raise RuntimeError('No profile passed validation; export disabled.')
    recipes=fresh(bank['test'],args.count,args.offset)
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:rows=render(pool,recipes,p)
    args.output.mkdir(parents=True,exist_ok=True);save_rows(args.output,'samples',rows)
    U.dump(args.output/'manifest.json',dict(family=args.family,count=args.count,offset=args.offset,selected_sha256=U.sha(dest/'selected.json'),complete=sum(r['complete'] for r in rows),usable=sum(r['usable'] for r in rows),errors=sum('error' in r for r in rows)))

if __name__=='__main__':main()
