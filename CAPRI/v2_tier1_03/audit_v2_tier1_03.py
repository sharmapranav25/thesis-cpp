from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).parent;seen={};seeds=set();result={}
for part in ('train','selection','test','heldout'):
    a=np.load(root/'v2_tier1_02'/(part+'_supervision.npz'))
    b=np.load(root/'v2_tier1_03'/(part+'_supervision.npz'))
    for key in ('images','full_masks','visible_masks','labels'):assert np.array_equal(a[key],b[key]),(part,key)
    assert np.all(b['visible_masks']<=b['full_masks'])
    records=json.loads((root/'v2_tier1_03'/(part+'_records.json')).read_text())
    for im,r in zip(b['images'],records):
        assert r['seed'] not in seeds;seeds.add(r['seed'])
        digest=hashlib.sha256(im.tobytes()).hexdigest();assert digest not in seen or seen[digest]==part;seen[digest]=part
    result[part]={'exact02_image_label_match':True,'visible_subset_full':True,'count':len(records)}
freshseeds=set()
for part in ('calibration','fresh_same','fresh_font','fresh_recipe'):
    path=root/'v2_tier1_03'/(part+'_records.json')
    if path.exists():
        for r in json.loads(path.read_text()):
            assert r['seed'] not in seeds and r['seed'] not in freshseeds;freshseeds.add(r['seed'])
result['fresh_seed_disjoint_count']=len(freshseeds)
(root/'v2_tier1_03/annotation_audit.json').write_text(json.dumps(result,indent=2))
print(result)
