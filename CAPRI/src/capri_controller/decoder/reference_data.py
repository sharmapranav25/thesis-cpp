"""Verified TRAIN font-library pixels, with library labels outside model inputs.

This is a reference reader, not a second corpus generation or a trained index.
Held-out galleries require a separate frozen evaluation protocol and are refused.
"""
import collections
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .common import read,sha,write,now
from .data import pixel_inputs,resize_geometry,box_targets


class TrainReferenceDataset(Dataset):
    def __init__(self,run):
        self.run=Path(run);root=self.run/'font_references_train_v1'
        protocol=read(root/'protocol.json');done=read(root/'completion.json')
        self.protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if protocol['split']!='train' or done['state']!='complete' or done['protocol_hash']!=self.protocol_hash:
            raise ValueError('Completed TRAIN references required')
        manifest=self.run/'fonts/font_manifest.json'
        if sha(manifest)!=protocol['font_manifest_sha256']:raise ValueError('Reference font identity mismatch')
        fonts={f['key']:f for f in read(manifest)['fonts'] if f['split']=='train'}
        plans=[p for p in (self.run/'font_reference_plans').glob('train_*.json') if sha(p)==protocol['plan_sha256']]
        if len(plans)!=1:raise ValueError('Unique hash-verified TRAIN reference plan required')
        plan=read(plans[0]);records=plan['records']
        if plan['font_manifest_sha256']!=protocol['font_manifest_sha256'] or len(records)!=protocol['total']:
            raise ValueError('Reference plan/manifest mismatch')
        self.rows=[];self.handles=collections.OrderedDict();archive_hashes={}
        for path in sorted(root.glob('shard_*.receipt.json')):
            receipt=read(path);archive=path.with_name(path.name.replace('.receipt.json','.zip'))
            if receipt['first']!=len(self.rows) or receipt['protocol_hash']!=self.protocol_hash or sha(archive)!=receipt['sha256']:
                raise ValueError('Reference archive continuity/identity mismatch')
            archive_hashes[str(archive)]=receipt['sha256']
            with zipfile.ZipFile(archive) as z:
                for row in receipt['records']:
                    record=records[len(self.rows)];font=fonts.get(record['font_key'])
                    if (record['specimen_id']!=row['specimen_id'] or not row['accepted'] or record['split']!='train'
                        or font is None or record['font_sha256']!=font['sha256'] or record['family_group']!=font['split_group']):
                        raise ValueError('Reference specimen/font provenance mismatch')
                    with z.open(row['stem']+'.png') as f:header=f.read(24)
                    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise ValueError('Invalid reference PNG')
                    width,height=struct.unpack('>II',header[16:24])
                    self.rows.append(dict(archive=str(archive),stem=row['stem'],pixel_sha256=row['pixel_sha256'],
                                          record=record,shape=resize_geometry(width,height)[2:]))
        if len(self.rows)!=len(records) or done['accepted_references']!=len(records):raise ValueError('Incomplete reference snapshot')
        self.snapshot_hash=hashlib.sha256(json.dumps(dict(protocol=self.protocol_hash,archives=archive_hashes,rows=self.rows),sort_keys=True).encode()).hexdigest()

    def __len__(self):return len(self.rows)

    def __getstate__(self):
        state=self.__dict__.copy();state['handles']=collections.OrderedDict();return state

    def close(self):
        for z in self.handles.values():z.close()
        self.handles.clear()

    def __getitem__(self,index):
        row=self.rows[index];path=row['archive']
        if path not in self.handles:
            self.handles[path]=zipfile.ZipFile(path)
            if len(self.handles)>4:self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(path);z=self.handles[path];stem=row['stem'];record=row['record']
        rgb=np.asarray(Image.open(io.BytesIO(z.read(stem+'.png'))).convert('RGB'))
        if hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()!=row['pixel_sha256']:
            raise ValueError('Reference pixel checksum mismatch')
        label=json.loads(z.read(stem+'.json'));glyphs=label['glyphs']
        if (label['recipe']['specimen_id']!=record['specimen_id'] or label['recipe']['split']!='train'
            or label['count']!=len(record['text']) or len(glyphs)!=label['count']):
            raise ValueError('Reference label identity mismatch')
        for i,g in enumerate(glyphs):
            if g['font_key']!=record['font_key'] or g['font_sha256']!=record['font_sha256'] or g['axes']!=record['axes'] or g['char']!=record['text'][i]:
                raise ValueError('Reference glyph provenance mismatch')
        inputs,geometry=pixel_inputs(rgb)
        # No text/count/family/box is included in model inputs. Library-only box
        # labels can audit predicted regions and associate them with known fonts.
        labels={k:record[k] for k in ('specimen_id','font_key','font_sha256','family_group','style','split','axes')}
        labels['boxes_visible']=box_targets(glyphs,'box_visible',geometry)
        return dict(inputs=inputs,library_labels=labels,audit=dict(stem=stem,geometry=geometry,pixel_sha256=row['pixel_sha256']))


def reference_batches(dataset,batch_size=8):
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size required')
    buckets=collections.defaultdict(list)
    for i,row in enumerate(dataset.rows):buckets[tuple(row['shape'])].append(i)
    return [indices[start:start+batch_size] for _,indices in sorted(buckets.items()) for start in range(0,len(indices),batch_size)]


def reference_collate(items):
    import torch
    if not items or len({tuple(x['inputs']['pixels'].shape) for x in items})!=1:raise ValueError('Exact-shape reference batch required')
    return dict(inputs={key:torch.stack([x['inputs'][key] for x in items]) for key in ('pixels','valid_pixels')},
                library_labels=[x['library_labels'] for x in items],audit=[x['audit'] for x in items])


def check_reference_data(run):
    import torch
    torch.set_num_threads(2);dataset=TrainReferenceDataset(run)
    try:
        batches=reference_batches(dataset)
        assert sorted(i for b in batches for i in b)==list(range(len(dataset)))
        selected=sorted(set([0,len(dataset)//2,len(dataset)-1]+[i for i in range(0,len(dataset),500)]))
        for i in selected:
            row=dataset[i]
            assert set(row['inputs'])=={'pixels','valid_pixels'}
            assert row['library_labels']['split']=='train' and 'text' not in row['library_labels']
            assert row['library_labels']['boxes_visible'].shape==(6,4)
            if not torch.isfinite(row['inputs']['pixels']).all():raise AssertionError('Nonfinite reference pixels')
        batch=reference_collate([dataset[i] for i in batches[0]])
        assert batch['inputs']['pixels'].shape[0]==len(batches[0])
        # Re-reading is stable; labels do not alter the shared pixel transform.
        first=dataset[0];again=dataset[0]
        assert all(torch.equal(first['inputs'][k],again['inputs'][k]) for k in first['inputs'])
        result=dict(created_at=now(),passed=True,split='train',references=len(dataset),snapshot_hash=dataset.snapshot_hash,
                    batches=len(batches),decoded_samples=len(selected),verified_archive_hashes=True,
                    all_samples_once=True,pixel_only_inputs=True,shared_aspect_preserving_transform=True,
                    source_sha256=sha(__file__),trained_embedding_index=False,
                    limitation='CPU reference-reader verification only. Predicted-region extraction and trained font-index creation require the later trained font branch; held-out galleries remain separate.')
        write(Path(run)/'reference_data_checks.json',result);return result
    finally:dataset.close()
