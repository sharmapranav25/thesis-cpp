"""Generic validation usability gate. Preserve preliminary search artifacts."""
from features9 import *
from build9 import render,panel
from run import load_images,save_rows
from concurrent.futures import ProcessPoolExecutor
import render7 as R
import argparse,copy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['sam','dinov3'],required=True);ap.add_argument('--checkpoint',type=Path);args=ap.parse_args()
    root=ROOT/args.backend;summary=U.read(root/'summary.json');validated={};model=None
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family,result in summary.items():
            dest=root/family;selection=U.read(dest/'selection.json')
            eligible=[r for r in selection if r['score']['complete']==1 and r['score']['usable']>=.9]
            # Neutral is always validated. If none qualify, explicitly return neutral with failure status.
            chosen=copy.deepcopy(min(eligible,key=lambda r:r['score']['objective']) if eligible else next(r for r in selection if r['index']==0))
            chosen['passed_usability_gate']=bool(eligible);U.dump(dest/'validated_selected.json',chosen)
            split=U.read(dest/'target_splits.json');recipes=U.read(dest/'recipe_bank.json');tests=load_images(split['test'])
            previous='controller' if chosen['index']==result['selected_index'] else 'baseline' if chosen['index']==0 else None
            if previous:
                labels=U.read(dest/previous/'labels.json');rows=[]
                for label in labels:
                    row=dict(label);row['image']=np.asarray(Image.open(dest/label['image']).convert('RGB'));row['mask']=np.asarray(Image.open(dest/label['mask']).convert('L'));rows.append(row)
                score=result['scores'][previous]
            else:
                if model is None:model=(SAM if args.backend=='sam' else DINOv3)(args.checkpoint)
                rows=render(pool,recipes['test'],chosen['controls']);tf=model.extract(tests);features=model.extract([r['image'] for r in rows])
                norm=U.read(dest/'normalization.json');obj=Objective.__new__(Objective);obj.std={k:np.asarray(v) for k,v in norm['std'].items()};obj.scale=norm['scale'];score=obj.score(features,tf,rows)
            save_rows(dest,'validated_controller',rows)
            baseline_labels=U.read(dest/'baseline/labels.json');base=[np.asarray(Image.open(dest/r['image']).convert('RGB')) for r in baseline_labels]
            panel(dest/'validated_renders.png',family+' | '+args.backend+' validated controller',['Original','Hybrid baseline','Baseline retained' if chosen['index']==0 else args.backend+' controller'],[[pixels(tests[i]),base[i],rows[i]['image']] for i in range(6)])
            validated[family]=dict(scores=dict(baseline=result['scores']['baseline'],controller=score),selected_index=chosen['index'],
                selected_sha256=U.sha(dest/'validated_selected.json'),passed_usability_gate=bool(eligible),retained_baseline=chosen['index']==0,
                preliminary_selected_index=result['selected_index'],candidates=result['candidates'],target_mask_coverage=result['target_mask_coverage'])
            U.dump(root/'validated_summary.json',validated);print(family,'validated selection',chosen['index'],'baseline retained',chosen['index']==0,flush=True)
    U.dump(root/'quality_gate_protocol.json',dict(rule='Among validation candidates, require complete=1 and usable>=0.90; choose lowest objective. If none qualify, return neutral and mark gate failure.',
        reason='Added after fresh City export exposed 4/8 usable preliminary renders. This is a development-stage revision; diagnostic target pixels are not a pristine blind test.',
        selection_data='Previously saved synthetic validation scores only. No diagnostic scores or images select the winner.',preliminary_artifacts_preserved=True))

if __name__=='__main__':main()
