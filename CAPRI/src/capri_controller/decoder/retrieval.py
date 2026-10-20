"""Replaceable visual font reference index; cosine scores are not probabilities.

Library-side labels associate predicted-region embeddings with known specimens.
Query inputs are visual embeddings and optional predicted quality weights only.
No target characters, family labels or requested count enter retrieval.
"""
import os
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import read, write, sha, now

IDENTITY_FIELDS = ('checkpoint_sha256', 'preprocessing_sha256', 'font_manifest_sha256',
                   'embedding_pipeline_sha256', 'weights', 'diagnostic')


def identity(value):
    if set(value) != set(IDENTITY_FIELDS): raise ValueError('Incomplete embedding identity')
    for key in IDENTITY_FIELDS[:4]:
        if not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{64}',value[key]):
            raise ValueError('Expected SHA256 for '+key)
    if value['weights'] not in ('raw','ema') or type(value['diagnostic']) is not bool:
        raise ValueError('Explicit raw/EMA and diagnostic identity required')
    return dict(value)


def vectors(value, allow_empty=False):
    x = torch.as_tensor(value).detach().to(device='cpu',dtype=torch.float32)
    if x.ndim!=2 or x.shape[1]!=128 or (not allow_empty and not len(x)):
        raise ValueError('Expected N x 128 font embeddings')
    if not torch.isfinite(x).all() or (x.norm(dim=-1)<1e-8).any():
        raise ValueError('Nonfinite or zero font embedding')
    return F.normalize(x,dim=-1)


class FontIndex:
    """Exact nearest-specimen search, grouped by alias-aware family.

    Maximum similarity preserves glyph/style modes; larger/more diverse banks
    can increase maxima. Record prototype counts and calibrate against the exact
    bank. Never interpret max cosine or an aggregate score as confidence.
    """
    def __init__(self, embeddings, records, embedding_identity):
        self.identity = identity(embedding_identity)
        self.embeddings = vectors(embeddings)
        self.records = [dict(r) for r in records]
        if len(records)!=len(self.embeddings): raise ValueError('Prototype/record count mismatch')
        seen=set(); files={}; families={}
        for i,r in enumerate(self.records):
            if not all(isinstance(r.get(k),str) and r[k] for k in ('prototype_id','font_key','font_sha256','family_group','style','split','specimen_id')):
                raise ValueError('Incomplete font specimen provenance')
            if r['prototype_id'] in seen: raise ValueError('Duplicate prototype identity')
            seen.add(r['prototype_id'])
            if not re.fullmatch('[0-9a-f]{64}',r['font_sha256']): raise ValueError('Invalid font hash')
            if r['font_key']!=r['font_sha256'][:20]: raise ValueError('Font key/hash mismatch')
            if r['split'] not in ('train','dev','calibration','test'): raise ValueError('Unknown font split')
            description=(r['font_sha256'],r['family_group'],r['style'],r['split'])
            if r['font_key'] in files and files[r['font_key']]!=description: raise ValueError('Inconsistent font provenance')
            files[r['font_key']]=description
            if r['family_group'] in families and families[r['family_group']]!=r['split']: raise ValueError('Family crosses splits')
            families[r['family_group']]=r['split']
        order=sorted(range(len(self.records)),key=lambda i:self.records[i]['prototype_id'])
        self.records=[self.records[i] for i in order];self.embeddings=self.embeddings[order]
        self.families=sorted(families)
        self.members={f:torch.tensor([i for i,r in enumerate(self.records) if r['family_group']==f]) for f in self.families}

    def search(self, embeddings, embedding_identity, topk=5, quality=None, excluded_families=()):
        if identity(embedding_identity)!=self.identity: raise ValueError('Query/index embedding identity mismatch')
        if type(topk) is not int or topk<1: raise ValueError('Positive integer topk required')
        x=vectors(embeddings,allow_empty=True); n=len(x)
        q=torch.ones(n) if quality is None else torch.as_tensor(quality,dtype=torch.float32,device='cpu')
        if q.shape!=(n,) or not torch.isfinite(q).all() or (q<0).any() or (q>1).any():
            raise ValueError('Quality must have one finite [0,1] value per query')
        excluded=set(excluded_families)
        families=[f for f in self.families if f not in excluded]
        if not n or not families or q.sum()<=0:
            return dict(status='no_visual_evidence' if not n or q.sum()<=0 else 'empty_candidate_bank',
                        per_query=[],aggregate=[],calibrated=False)
        # Chunk query rows to bound the largest similarity matrix.
        scores=[]; winners=[]
        for chunk in x.split(32):
            per_family=[]; per_winner=[]
            for f in families:
                ids=self.members[f];best=torch.full((len(chunk),),-torch.inf); winner=torch.zeros(len(chunk),dtype=torch.long)
                for sub in ids.split(512):
                    similarity=(chunk@self.embeddings[sub].T).clamp(-1,1)
                    values,indices=similarity.max(-1);improved=values>best
                    best=torch.where(improved,values,best);winner=torch.where(improved,sub[indices],winner)
                per_family.append(best);per_winner.append(winner)
            scores.append(torch.stack(per_family,-1));winners.append(torch.stack(per_winner,-1))
        score=torch.cat(scores);winner=torch.cat(winners)
        def ranking(row):
            return sorted(range(len(families)),key=lambda j:(-float(row[j]),families[j]))[:topk]
        per_query=[];votes=torch.zeros(len(families))
        for i,row in enumerate(score):
            order=ranking(row)
            tied=torch.nonzero(row==row[order[0]]).flatten();votes[tied]+=q[i]/len(tied)
            per_query.append(dict(query_index=i,quality=float(q[i]),candidates=[dict(
                family_group=families[j],similarity=float(row[j]),
                font_key=self.records[int(winner[i,j])]['font_key'],
                prototype_id=self.records[int(winner[i,j])]['prototype_id']) for j in order]))
        pooled=(score*q[:,None]).sum(0)/q.sum();votes/=q.sum()
        aggregate=[dict(family_group=families[j],mean_similarity=float(pooled[j]),
                        best_match_weight_fraction=float(votes[j]),prototype_count=len(self.members[families[j]]))
                   for j in ranking(pooled)]
        return dict(status='uncalibrated_candidates',per_query=per_query,aggregate=aggregate,calibrated=False,
                    scoring='max specimen cosine per family; quality-weighted mean across queries',
                    limitation='Nearest available candidate is not verified font identity. Per-query candidates preserve mixture evidence; aggregate assumes shared-family compatibility. Bank size affects maxima.')

    def save(self, destination):
        destination=Path(destination)
        if destination.exists(): raise FileExistsError('Preserving existing font index')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.writing.'+str(os.getpid())+'.'+str(time.time_ns()))
        temporary.mkdir()
        with (temporary/'embeddings.npy').open('wb') as f:
            np.save(f,self.embeddings.numpy(),allow_pickle=False);f.flush();os.fsync(f.fileno())
        write(temporary/'metadata.json',dict(schema='capri-font-index-v1',identity=self.identity,records=self.records))
        receipt=dict(created_at=now(),count=len(self.records),families=len(self.families),
                     files={name:sha(temporary/name) for name in ('embeddings.npy','metadata.json')})
        write(temporary/'receipt.json',receipt)
        loaded=FontIndex.load(temporary,self.identity,allow_diagnostic=True)
        if not torch.allclose(loaded.embeddings,self.embeddings,atol=2e-7,rtol=2e-7): raise ValueError('Index verification failed')
        temporary.rename(destination)
        return receipt

    @classmethod
    def load(cls, folder, expected_identity, allow_diagnostic=False):
        folder=Path(folder);receipt=read(folder/'receipt.json')
        if set(receipt['files'])!={'embeddings.npy','metadata.json'}: raise ValueError('Invalid index receipt')
        if any(sha(folder/name)!=digest for name,digest in receipt['files'].items()): raise ValueError('Font index hash mismatch')
        metadata=read(folder/'metadata.json')
        if metadata['schema']!='capri-font-index-v1': raise ValueError('Unsupported font index schema')
        if identity(expected_identity)!=metadata['identity']: raise ValueError('Font index identity mismatch')
        if metadata['identity']['diagnostic'] and not allow_diagnostic: raise ValueError('Diagnostic font index cannot serve production')
        result=cls(np.load(folder/'embeddings.npy',allow_pickle=False),metadata['records'],metadata['identity'])
        if len(result.records)!=receipt['count'] or len(result.families)!=receipt['families']: raise ValueError('Font index count mismatch')
        return result
