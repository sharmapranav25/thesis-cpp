"""Separate single-cue layout probes from the algebraically redundant joint probe."""
from pathlib import Path
import sys,json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'phase3_reader'))
from common import CORPUS,read,write
data=BASE/'phase3_city_data_01';rows=[json.loads(l) for l in (data/'partitions.jsonl').open()];x=[];y=[]
with (CORPUS/'manifest.jsonl').open() as f:
    for line in f:
        r=json.loads(line);boxes=np.array([g['box'] for g in r['recipe']['glyphs']]);centers=(boxes[:,0]+boxes[:,2])/2
        # Image width from corpus layout operations is not inferred from labels; read image header only.
        from PIL import Image
        with Image.open(CORPUS/r['filename']) as im:w,h=im.size
        x.append([float(np.diff(centers).mean()/w),float((boxes[:,2]-boxes[:,0]).mean()/w),float((boxes[:,3]-boxes[:,1]).mean()/h)])
        y.append(r['length'])
x=np.asarray(x);y=np.asarray(y);train=np.array([r['partition']=='train' for r in rows]);test=np.array([r['partition']=='test' for r in rows]);results={}
for j,name in enumerate(['normalized_mean_center_gap_only','normalized_mean_glyph_width_only','normalized_mean_glyph_height_only']):
    m=ExtraTreesClassifier(n_estimators=50,max_depth=8,n_jobs=2,random_state=731).fit(x[train,j:j+1],y[train]);results[name]=float(np.mean(m.predict(x[test,j:j+1])==y[test]))
report=dict(single_feature_test_accuracy=results,limitation='The earlier joint occupied-center-span + mean-center-gap probe encodes the identity n=span/mean_gap+1; its100% result alone is not independent evidence. Single-cue probes avoid that pair. Cues still use synthetic ground-truth glyph geometry, not image detections, so they diagnose renderer priors rather than prove what the image reader learned.',metadata_only=True,real_labels_used=False)
write(BASE/'phase3_shortcut_followup.json',report);print(json.dumps(report,indent=2))
