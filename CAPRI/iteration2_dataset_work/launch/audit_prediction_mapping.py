import sys,json,gzip,copy,time,argparse,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT.parent/'captcha-effects'))
from captcha_effects import text as T,render as R
from captcha_effects.effects import REGISTRY
from prediction_render_mapping import map_recipe
root=Path('C:/Users/pranav/Documents/ChatGPT/thesis')
parser=argparse.ArgumentParser();parser.add_argument('--out',default='prediction_mapping_audit_01');parser.add_argument('--oracle-n',type=int,default=8);args=parser.parse_args()
out=root/args.out;out.mkdir(exist_ok=False)
shutil.copy2(Path(__file__).parent/'prediction_render_mapping.py',out/'mapping_snapshot.py')
bank=T.FontBank();summary=defaultdict(list)
# Quantify the old mapping on every full-set output already generated.
for line in (root/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines():
    row=json.loads(line)
    for h,g in row['effects'].items():
        if g['present']:
            actual=row['actual_rendered_effects'][h]
            summary[h].append(abs(g['level']-actual['level']))
full_audit={h:dict(n=len(v),mean_absolute_level_error=float(np.mean(v)),within_one=float(np.mean(np.array(v)<=1))) for h,v in summary.items()}
(out/'old_full_set_effect_mapping.json').write_text(json.dumps(full_audit,indent=2))
real=json.loads((root/'iter2_target_recreation_01/report.json').read_text())['results']
results=[];start=time.perf_counter()
for i,row in enumerate(real):
    original=Image.open(row['source']).convert('RGB')
    mapped=map_recipe(bank,original.size,row['effects_spec'],row['text_spec'],row,seed=9000+i)
    prefix=f'{row["family"]}_{i:02d}'
    pictures=[original,Image.fromarray(mapped['before']['image']),Image.fromarray(mapped['after']['image'])]
    scale=2 if original.width>250 else 3
    panel=Image.new('RGB',(max(650,original.width*scale),3*(original.height*scale+35)+20),'white');d=ImageDraw.Draw(panel)
    titles=['REAL TARGET','OLD MAPPING (controlled seed)','CORRECTED MAPPING (same seed/predictions/text)']
    for j,(im,title) in enumerate(zip(pictures,titles)):
        y=j*(original.height*scale+35);d.text((8,y+5),title,fill='black');panel.paste(im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST),(0,y+25))
    panel.save(out/f'{prefix}.png')
    record=dict(family=row['family'],source=row['source'],before_distance=mapped['before_distance'],after_distance=mapped['after_distance'],changes=mapped['history'],calls=mapped['calls'],
                effects_spec=mapped['effects_spec'],text_spec=mapped['text_spec'],before_measured=dict(effects=mapped['before']['effects'],text=mapped['before']['text']['heads']),after_measured=dict(effects=mapped['after']['effects'],text=mapped['after']['text']['heads']),comparison=str(out/f'{prefix}.png'))
    results.append(record);print(json.dumps({k:record[k] for k in ('family','before_distance','after_distance','calls')}),flush=True)
# Oracle check: synthetic ground-truth features isolate mapping from model error.
label_path=PROJECT/'runs/iter2_baseline_15_20261003_04/iter2_h96/seen_test/labels.jsonl.gz'
oracle=[]
with gzip.open(label_path,'rt') as f:
    for j,line in enumerate(f):
        if j>=args.oracle_n:break
        lab=json.loads(line);ts=copy.deepcopy(lab['text_spec'])
        goals=dict(effects=lab['effects'],text_features=lab['text']['heads'])
        es=dict(background=lab['background'],effects=[])
        for h,g in lab['effects'].items():
            if g['present']:es['effects'].append(dict(name=g['effect'],level=g['level']))
        # Deliberately use the old formula for oracle geometry: this tests the
        # translation with correct features, independently of learned weights.
        for h,key,base,step in [('per_char_rotation','rotation',3,4.2),('vertical_jitter','jitter',0,.015),('size_variation','size_jitter',0,.04),('text_on_curve','curve',0,.02),('overlap_tight_spacing','spacing',0,-.025),('wide_spacing','spacing',0,.1)]:
            g=lab['text']['heads'][h]
            if g['present']:ts[key]=base+step*g['level']
        result=map_recipe(bank,tuple(lab['size']),es,ts,goals,seed=9500+j)
        oracle.append(dict(index=j,before_distance=result['before_distance'],after_distance=result['after_distance'],changes=result['history']))
        print(f'Oracle mapping check {j+1}/{args.oracle_n}',flush=True)
report=dict(real=results,oracle=oracle,seconds=time.perf_counter()-start,
            oracle_before_mean=float(np.mean([r['before_distance'] for r in oracle])),oracle_after_mean=float(np.mean([r['after_distance'] for r in oracle])),
            note='No model changes. Fixed predicted descriptions. Renderer measured-label objective only; no real answer labels, target-pixel matching, colour fitting, or family rules. Active binary/ordinal contradictions remain visible. Different colours/positions are unsupported outputs, not solved by this correction.')
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('real','oracle')}),flush=True)
