"""All eight current sources: Phase1 directly to frozen font inference."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load,pixel_hash,digest
from style_pipeline.font_match import FontMatcher,infer
from style_pipeline.measure import extract_style
from render_color_locked import render
from style_pipeline.recipes import apply_recipe,RecipeCNN

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
out=new_run(Path(__file__).resolve().parent/'phase3_parameter_audit_01');torch.set_num_threads(4)
run=ROOT/'runs/style_phase3_font_05';fonts=read(run/'fonts.json');checkpoint=run/'font_matcher.pt';before=digest(checkpoint)
model=FontMatcher(len(fonts)).cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);model.eval()
families={};variants={};old=read(ROOT/'runs/style_phase1_05/results.json')['families']
for name in old:
    folder=ROOT/'runs'/('style_frozen_'+name+('_02' if name=='darkbay' else '_01'))
    protocol=read(folder/'protocol.json')
    for f,p in protocol['models'].items():assert digest(p)==protocol['model_sha256_before'][f]
    lookup={r['pixel_sha256']:r for r in jsonl(folder/'predictions.jsonl')}
    rows=read(ROOT/'runs/style_phase1_05'/f'{name}_measurements.json')['fit'][:16]
    families[name]=[lookup[r['pixel_sha256']] for r in rows];variants[name]=old[name]['selected']['variant']
lookup={r['pixel_sha256']:r for r in jsonl(ROOT/'runs/style_frozen_inference_01/predictions.jsonl')}
families['roncuevas']=[lookup[r['pixel_sha256']] for r in read(ROOT/'runs/style_phase1_unseen_01/split.json')['fit'][:16]]
for name,folder in [('pymlex','style_phase3_new_family_01'),('sloth','style_phase3_new_family_02')]:families[name]=[r for r in read(ROOT/'runs'/folder/'predictions.json') if r['partition']=='train'][:16]

from style_pipeline.refiners import fixed_features
prior=read(ROOT/'runs/style_direct_bank_01/results.json');controls=read(ROOT/'runs/style_phase3_color_locked_01/results.json')
params=['text_height','tilt','spacing','line_count','line_width','dot_count','curve_count','curve_amplitude']
def features(im):
    a=np.asarray(im.convert('L').resize((192,64)),np.float32)/255
    t=torch.from_numpy(a)[None,None].repeat(1,3,1,1)
    return fixed_features(t).numpy()[0]
summary={};failures=[]
for name,rows in families.items():
    refs=[load(r['path']) for r in rows[:4]];target=np.mean([features(im) for im in refs],axis=0)
    chosen=next(i for i,f in enumerate(fonts) if f['family']==prior[name]['font']);p=np.array(controls[name]['parameters']);allims={};records={}
    methods=[('baseline',None)]+[(key,j+3) for j,key in enumerate(params)]+[('all_noncolor',-1)]
    for method,j in methods:
        vector=np.zeros(11)
        if j==-1:vector=p.copy()
        elif j is not None:vector[j]=p[j]
        ims=[];distances=[];changed=[]
        for i,(im,r) in enumerate(zip(refs,rows)):
            style=extract_style(im);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()};s=apply_recipe(style,vector);s['font_effect']=prior[name]['effect'];br=prior[name]['recipes'][i]
            for key in ('background_palette','foreground_palette','gradient_left','gradient_right'):s[key]=copy.deepcopy(style[key])
            s['_locked_operation_colors']={kind:[o['color'] for o in br['operations'] if o['kind']==kind] for kind in ('block','line','point','classifier_curve')};s['_locked_glyph_colors']=[g['color'] for g in br['glyphs']]
            try:
                final,_,recipe=render(s,br['text'],fonts[chosen]['path'],br['seed'],br['variant'])
                assert [g['color'] for g in recipe['glyphs']]==s['_locked_glyph_colors']
                for op in recipe['operations']:assert op['color'] in s['_locked_operation_colors'][op['kind']]
            except ValueError as exc:failures.append(dict(family=name,method=method,index=i,error=str(exc)));continue
            final.save(out/f'{name}_{method}_{i}.png');ims.append(final)
            base=load(ROOT/'runs/style_direct_bank_01'/f'{name}_{i:02d}.png');changed.append(not np.array_equal(np.asarray(base),np.asarray(final)))
            if method=='baseline':assert not changed[-1]
            distances.append(float(np.abs(features(final)-features(im)).mean()))
        allims[method]=ims
        records[method]=dict(value=float(p[j]) if j is not None and j>=0 else None,rendered=len(ims),pixel_changed=sum(changed),mean_pair_distance=float(np.mean(distances)) if distances else None,distribution_distance=float(np.abs(np.mean([features(im) for im in ims],axis=0)-target).mean()) if ims else None)
    base=records['baseline']['distribution_distance'];pairbase=records['baseline']['mean_pair_distance']
    for method,r in records.items():
        r['distribution_improvement_pct']=100*(1-r['distribution_distance']/base);r['pair_improvement_pct']=100*(1-r['mean_pair_distance']/pairbase)
    summary[name]=records
    panel=Image.new('RGB',(1000,1540),'white');d=ImageDraw.Draw(panel)
    for rn,(method,ims) in enumerate([('reference',refs)]+list(allims.items())):
        desc=method if method=='reference' else f"{method} | distance improvement {records[method]['distribution_improvement_pct']:.1f}%"
        d.text((10,rn*140+5),name+' | '+desc,fill='black')
        for i,im in enumerate(ims):
            im=im.copy();im.thumbnail((235,100));panel.paste(im,(10+i*250,rn*140+30))
    panel.save(out/f'{name}_all_controls.png')
write(out/'results.json',summary);write(out/'failures.json',failures)
print({name:{k:round(v['distribution_improvement_pct'],1) for k,v in r.items() if v['pixel_changed']} for name,r in summary.items()})
