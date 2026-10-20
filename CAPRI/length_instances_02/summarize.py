from pathlib import Path
import json,collections
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());lines=['# Character count and instance comparison — results','', 'Synthetic metrics only. Real outputs are unlabelled predictions, not accuracy.','', '| Model | Seed | Selected step | Same exact | Font exact | Layout exact |','|---|---:|---:|---:|---:|---:|']
for name,row in r.items():
 for output in ['count_head','instances']:
  if output not in row['tests']['test']:continue
  values=[row['tests'][key][output]['exact'] for key in ['test','font_test','layout_test']]
  lines.append('| '+name.rsplit('_',1)[0]+' / '+output+' | '+name.rsplit('_',1)[1]+' | '+str(row['selected_step'])+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
lines+=['','## Layout shortcut audit','','A metadata-only ExtraTrees baseline uses requested occupied span and average requested glyph width, not image pixels. This is a diagnostic for residual selection bias, not a proposed real-image estimator. Fit on synthetic training; no choices based on locked tests.']
primary=[]
for name,row in r.items():
 for test in ['test','font_test','layout_test']:
  z=np.load(ROOT/f'{name}_{test}.npz');mask=(z['truth']>=2)&(z['truth']<=8);y=z['truth'][mask];pred=z['probabilities'][mask].argmax(1);record=dict(model=name,test=test,output='count_head',exact=float(np.mean(pred==y)),mae=float(np.abs(pred-y).mean()))
  primary.append(record)
  if row['object_threshold'] is not None:
   pred=(z['object_probabilities'][mask]>=row['object_threshold']).sum(1);primary.append(dict(model=name,test=test,output='instances',exact=float(np.mean(pred==y)),mae=float(np.abs(pred-y).mean())))
(ROOT/'primary_range_metrics.json').write_text(json.dumps(primary,indent=2))
def features(name):
 rows=json.loads((ROOT/(name+'_metadata.json')).read_text());rows=[x for x in rows if 2<=x['n']<=8];return np.array([[x['span'],x['mean_requested_width']] for x in rows]),np.array([x['n'] for x in rows])
x,y=features('train');model=ExtraTreesClassifier(n_estimators=100,max_depth=4,random_state=17).fit(x,y);audit={}
for name in ['test','font_test','layout_test']:
 x,y=features(name);pred=model.predict(x);audit[name]=dict(exact=float(np.mean(pred==y)),chance=1/7,span_count_correlation=float(np.corrcoef(x[:,0],y)[0,1]),mean_width_count_correlation=float(np.corrcoef(x[:,1],y)[0,1]));lines.append(f'- {name}: metadata-only exact {audit[name]["exact"]:.3f}; uniform chance {1/7:.3f}; span/count correlation {audit[name]["span_count_correlation"]:.3f}.')
(ROOT/'shortcut_audit.json').write_text(json.dumps(audit,indent=2))
realpath=ROOT/'real_inference.json'
if realpath.exists():
 real=json.loads(realpath.read_text());lines+=['','## Real style-fit predictions','','Counts below are ensemble modal predictions. They are not reference labels. Disagreement and histograms are retained rather than forcing every image to one family length.']
 out={}
 for family,value in real.items():
  out[family]={};lines+=['','### '+family,'']
  for kind in ['count_cnn','instance_transformer']:
   models=[m for m in value['models'] if m['model']==kind];probs=np.array([m['count_probabilities'] for m in models]);pred=probs.mean(0).argmax(1);individual=probs.argmax(2);hist=dict(collections.Counter(map(int,pred)));agree=float(np.mean(np.all(individual==individual[0],axis=0)));out[family][kind]=dict(images=len(pred),predicted_histogram=hist,three_seed_agreement=agree);lines.append(f'- {kind}: {hist}; three-seed exact agreement {agree:.1%}.')
   if kind=='instance_transformer':
    counts=np.array([m['instance_counts'] for m in models]);out[family][kind]['instance_median_histogram']=dict(collections.Counter(map(int,np.median(counts,axis=0))));lines.append('- Instance-branch median histogram: '+str(out[family][kind]['instance_median_histogram']))
 (ROOT/'real_summary.json').write_text(json.dumps(out,indent=2))
lines+=['','## Decision','','No automatic pipeline promotion. Synthetic scores and real seed agreement do not establish real accuracy. Review visible instance locations, short-length errors, and independent count-only evaluation before changing generation lengths. The existing2–8 policy remains until that evidence is available.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines[:18]))
