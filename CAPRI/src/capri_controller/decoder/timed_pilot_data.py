"""Immutable, paired family/count-stratified diagnostic subsets."""
import collections
import hashlib
import json
import zipfile
from torch.utils.data import Dataset
from .common import write


class PilotSubset(Dataset):
    def __init__(self,base,indices):
        self.base=base;self.indices=list(indices);self.rows=[base.rows[i] for i in indices]
        self.split=base.split;self.family_indices=base.family_indices;self.fonts=base.fonts
        self.snapshot_hash=hashlib.sha256(json.dumps(dict(base=base.snapshot_hash,indices=self.indices),sort_keys=True).encode()).hexdigest()
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.base[self.indices[index]]
    def close(self):self.base.close()


def subset(base,count,seed):
    if count%2 or count>len(base):raise ValueError('Complete paired subset required')
    families=len({f['family_group'] for f in base.fonts.values() if f['split']==base.split})
    lookup={int(row['stem'].rsplit('_',1)[1]):i for i,row in enumerate(base.rows)}
    buckets=collections.defaultdict(list)
    for number,i in lookup.items():
        if number%2:continue
        if number+1 not in lookup:raise ValueError('Incomplete pair')
        group=number//2
        rank=hashlib.sha256(f'{seed}/{base.split}/{group}'.encode()).hexdigest()
        buckets[((group//7)%families,group%7)].append((rank,i,lookup[number+1]))
    for rows in buckets.values():rows.sort()
    order=sorted(buckets,key=lambda k:hashlib.sha256(f'{seed}/{k}'.encode()).hexdigest())
    selected=[];cycle=0
    while len(selected)<count:
        for key in order:
            if len(selected)==count:break
            if cycle<len(buckets[key]):selected.extend(buckets[key][cycle][1:])
        cycle+=1
    return PilotSubset(base,sorted(selected))


def audit(dataset,destination):
    result={k:collections.Counter() for k in ('count','family','layout','effect','intensity','pair_kind','multicolour')}
    handles={}
    try:
        for row in dataset.rows:
            if row['archive'] not in handles:handles[row['archive']]=zipfile.ZipFile(row['archive'])
            label=json.loads(handles[row['archive']].read(row['stem']+'.json'));recipe=label['recipe']
            assert recipe['split']==dataset.split
            result['count'][str(label['count'])]+=1;result['layout'][recipe['text']['layout']]+=1
            result['pair_kind'][str(recipe['pair_kind'])]+=1;result['multicolour'][str(recipe['requested_multicolour'])]+=1
            for key in set(recipe['text']['font_keys']):result['family'][dataset.fonts[key]['split_group']]+=1
            for key,effect in label['effects'].items():
                if effect['present']:result['effect'][key]+=1;result['intensity'][f'{key}/{effect["level"]}']+=1
    finally:
        for z in handles.values():z.close()
    value=dict(split=dataset.split,images=len(dataset),snapshot_hash=dataset.snapshot_hash,indices=dataset.indices,
        policy='Hash-ranked paired units, round-robin original quota-family/count anchors; effects/layout/colour audited, not jointly quota-balanced.',
        distributions={k:dict(sorted(v.items())) for k,v in result.items()})
    write(destination,value);return value
