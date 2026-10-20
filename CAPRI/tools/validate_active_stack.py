"""Regression evidence: run legacy and maintained implementations in separate processes."""
from pathlib import Path
import argparse,hashlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

WORK=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('side',choices=['legacy','current']);args=p.parse_args();start=time.time()
    if args.side=='legacy':
        sys.path.insert(0,str(WORK/'capri_darkbay_workflow'))
        from pipeline import U,R,Mixed,draw_recipe,load_images
    else:
        sys.path.insert(0,str(WORK/'src'))
        from capri_controller.pipeline import U,R,Mixed,draw_recipe,load_images
    out=WORK/'maintenance/validation'/args.side;out.mkdir(parents=True,exist_ok=True)
    indices=[0,50,99];entries=U.read(WORK/'capri_darkbay_workflow/entries.json');refs=U.read(WORK/'capri_darkbay_workflow/inputs.json')
    targets=load_images([refs[i] for i in indices]);model=Mixed()
    descriptions=model.custom.describe(targets);features=model.extract(targets)
    np.savez_compressed(out/'features.npz',**{f'{j}_{k}':v for j,part in enumerate(features) for k,v in part.items()})
    jobs=[(draw_recipe(e['recipe'],7200000000+int(e['input_index'])),e['controls']) for e in entries if e['input_index'] in indices]
    with ProcessPoolExecutor(max_workers=2,initializer=R.install) as pool:rows=list(pool.map(R.safe_render,jobs))
    for row in rows:
        if 'error' in row:raise RuntimeError(row['error'])
    hashes=[hashlib.sha256(row['image'].tobytes()).hexdigest() for row in rows]
    legacy_imports=sorted({str(Path(m.__file__).resolve()) for m in sys.modules.values() if getattr(m,'__file__',None) and any(s in str(m.__file__) for s in ['controller_mvp_','learned_controller_20261004'])})
    if args.side=='current':assert not legacy_imports,legacy_imports
    U.dump(out/'result.json',dict(descriptions=descriptions,render_hashes=hashes,texts=[r['text'] for r in rows],complete=[r['complete'] for r in rows],
        usable=[r['usable'] for r in rows],legacy_imports=legacy_imports,seconds=time.time()-start))
    print(args.side,'verified',len(targets),'inferences and',len(rows),'multiprocess renders')

if __name__=='__main__':main()
