"""Stateless TRAIN reference choices tied to synthetic scene families.

No prefetch/sampling cursor or global RNG is consumed. A phase must freeze this
contract and supply its epoch, reference budget and pairing mode explicitly.
"""
import collections
import hashlib
import json
from .common import sha
from .reference_data import reference_collate


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class ReferenceSampler:
    def __init__(self,dataset,family_indices,seed):
        if type(seed) is not int:raise ValueError('Explicit integer reference seed required')
        if (not family_indices or any(type(v) is not int or v<0 for v in family_indices.values())
            or len(set(family_indices.values()))!=len(family_indices)):
            raise ValueError('Preserve the unique original corpus family indices')
        self.dataset=dataset;self.family_indices=dict(family_indices)
        self.by_family=collections.defaultdict(list);self.font_families={};seen=set()
        for i,row in enumerate(dataset.rows):
            record=row['record'];family=record['family_group'];font=record['font_key']
            if (record['split']!='train' or family not in family_indices or record['specimen_id'] in seen
                or len(row['shape'])!=2 or any(type(v) is not int or v<1 for v in row['shape'])):
                raise ValueError('Unique verified TRAIN reference records/shapes required')
            if font in self.font_families and self.font_families[font]!=family:
                raise ValueError('Font key assigned to inconsistent families')
            seen.add(record['specimen_id']);self.font_families[font]=family;self.by_family[family].append(i)
        if not seen:raise ValueError('Empty TRAIN reference snapshot')
        self.contract=dict(schema='capri-reference-sampling-v1',seed=seed,reference_snapshot=dataset.snapshot_hash,
            family_indices_sha256=digest(family_indices),source_sha256=sha(__file__),
            policy='one specimen per selected scene family; hash-ranked families and eligible specimens; exact-shape subbatches')
        self.contract_hash=digest(self.contract)

    def select(self,scene_batch,epoch,max_references,mode):
        if type(epoch) is not int or epoch<1 or type(max_references) is not int or max_references<1:
            raise ValueError('Explicit positive epoch and reference budget required')
        if mode not in ('family','font_file'):raise ValueError('Explicit family or font_file pairing mode required')
        targets=scene_batch['targets'];audits=scene_batch['audit']
        if not targets or len(targets)!=len(audits):raise ValueError('Scene supervision/audit mismatch')
        present=collections.defaultdict(set);identity=[]
        for target,audit in zip(targets,audits):
            ids=target['font_family'].tolist();keys=audit['font_keys']
            if len(ids)!=len(keys) or not ids or not isinstance(audit['stem'],str) or not audit['stem']:
                raise ValueError('Scene font audit mismatch')
            for index,font in zip(ids,keys):
                family=self.font_families.get(font)
                if family is None or index!=self.family_indices[family]:
                    raise ValueError('Scene font file/family differs from verified TRAIN mapping')
                present[family].add(font)
            identity.append(dict(stem=audit['stem'],fonts=sorted(zip(keys,ids))))
        identity=sorted(identity,key=lambda v:(v['stem'],v['fonts']))
        context=dict(contract_hash=self.contract_hash,epoch=epoch,max_references=max_references,mode=mode,scenes=identity)
        context_hash=digest(context)
        ranked=sorted(present,key=lambda family:(digest([context_hash,'family',family]),family))
        selected=[]
        for family in ranked[:max_references]:
            candidates=[i for i in self.by_family[family]
                        if mode=='family' or self.dataset.rows[i]['record']['font_key'] in present[family]]
            if not candidates:raise ValueError('Missing eligible reference for selected TRAIN family')
            index=min(candidates,key=lambda i:(digest([context_hash,'specimen',self.dataset.rows[i]['record']['specimen_id']]),i))
            row=self.dataset.rows[index];record=row['record']
            selected.append(dict(index=index,specimen_id=record['specimen_id'],font_key=record['font_key'],
                                 family_group=family,shape=list(row['shape'])))
        result=dict(context=context,selected=selected,available_families=sorted(present),
                    omitted_families=sorted(set(present)-{r['family_group'] for r in selected}),
                    references=len(selected),reference_snapshot=self.dataset.snapshot_hash)
        result['sha256']=digest(result)
        return result

    def materialize(self,selection):
        if (digest({k:v for k,v in selection.items() if k!='sha256'})!=selection['sha256']
            or selection['context']['contract_hash']!=self.contract_hash
            or selection['reference_snapshot']!=self.dataset.snapshot_hash):
            raise ValueError('Reference choice/contract/snapshot changed')
        records=selection['selected']
        if (selection['references']!=len(records) or not records
            or len({r['index'] for r in records})!=len(records)):
            raise ValueError('Missing/repeated selected reference')
        buckets=collections.defaultdict(list)
        for record in records:
            index=record['index']
            if type(index) is not int or not 0<=index<len(self.dataset.rows):raise ValueError('Invalid reference index')
            row=self.dataset.rows[index];known=row['record']
            if (any(record[k]!=known[k] for k in ('specimen_id','font_key','family_group'))
                or record['shape']!=list(row['shape'])):raise ValueError('Selected reference identity changed')
            buckets[tuple(row['shape'])].append(index)
        return [reference_collate([self.dataset[i] for i in indices]) for _,indices in sorted(buckets.items())]
