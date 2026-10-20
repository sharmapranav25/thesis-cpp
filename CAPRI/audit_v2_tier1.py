from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
root=Path(__file__).parent/'v2_tier1_02';seen={};seeds=set();results={}
for part in ('train','selection','test','heldout'):
    d=np.load(root/(part+'_supervision.npz'));records=json.loads((root/(part+'_records.json')).read_text());assert len(records)==len(d['images']);assert np.all(d['visible_masks']<=d['full_masks']);assert np.isfinite(d['images']).all()
    for im,record in zip(d['images'],records):
        assert record['seed'] not in seeds;seeds.add(record['seed']);h=hashlib.sha256(im.tobytes()).hexdigest()
        if h in seen:assert seen[h]==part
        seen[h]=part
    positives=d['labels'].sum(0).tolist();results[part]=dict(count=len(records),positive_counts=positives,visible_support_subset_full=True)
    if part=='train':
        canvas=Image.new('RGB',(1000,720),'white');draw=ImageDraw.Draw(canvas)
        for row,i in enumerate([0,1,2,3,4,5]):
            im=Image.fromarray(d['images'][i]);canvas.paste(im,(5,20+row*115));draw.text((5,5+row*115),f'seed {records[i]["seed"]}',fill='black')
            for col,name in enumerate(['line','curve','dot','rectangle','grain']):
                m=d['visible_masks'][i,col];overlay=im.copy();color=Image.new('RGB',im.size,(255,20,20));overlay=Image.composite(color,im,Image.fromarray(m));canvas.paste(overlay,(150+col*165,20+row*115));draw.text((150+col*165,5+row*115),name,fill='black')
        canvas.save(root/'supervision_preview.png')
(root/'annotation_audit.json').write_text(json.dumps(dict(splits=results,seed_disjoint=True,pixels_disjoint_across_partitions=True,limitation='visible masks represent compositing support; not a perceptual counterfactual contrast audit'),indent=2));print(results)
