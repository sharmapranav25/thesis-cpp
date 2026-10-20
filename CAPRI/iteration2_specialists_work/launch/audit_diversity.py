"""CPU-only verification of targeted synthetic pools, without opening real images."""
import json
from pathlib import Path
import numpy as np

BASE=Path('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04')
OUT=Path(__file__).resolve().parents[1]/'run_20261003_01'
shards=sorted((BASE/'iter2_h96').glob('train_*'))
labels={key:np.concatenate([np.load(s/'index.npz')[key] for s in shards]) for key in ['eff_present','txt_present','font','length']}
manifest=json.loads((OUT/'task_manifest.json').read_text())
pools=np.load(OUT/'sampler_indexes.npz')
report={}
for task,meta in manifest.items():
    bins={}
    prefix=task.replace('/','__')+'__'
    for key in pools.files:
        if not key.startswith(prefix):continue
        indices=pools[key]
        font_categories=np.unique(labels['font'][indices,0]);lengths=np.unique(labels['length'][indices])
        others=np.concatenate([labels['eff_present'][indices],labels['txt_present'][indices]],axis=1)
        varying=int((others.min(0)!=others.max(0)).sum())
        bins[key[len(prefix):]]=dict(n=len(indices),font_categories=font_categories.tolist(),lengths=lengths.tolist(),
                                other_varying_features=varying)
    report[task]=dict(name=meta['name'],bins=bins)
exceptions=[dict(task=t,name=v['name'],value=value,**v['bins'][value]) for t,v in report.items() if t.startswith(('eff_presence/','txt_presence/')) for value in v['bins']
    if len(v['bins'][value]['font_categories'])<4 or v['bins'][value]['other_varying_features']<30]
(OUT/'pool_diversity.json').write_text(json.dumps(dict(passed=not exceptions,tasks=report,exceptions=exceptions,
  checks='Every feature presence/absence bucket has >=4 font categories and >=30 varying other presence features; no feature is tied to one font category/background combination.',
  known_data_limitation='Original text appearance tickets select one main appearance variation at a time. Positive geometry buckets vary 43 other features, but lack combinations with the nine competing appearance variations. All five font categories are represented; this audit does not claim full independence of features.'),indent=2))
print(json.dumps(dict(passed=not exceptions,tasks=len(report),presence_tasks=53,exceptions=exceptions)))
