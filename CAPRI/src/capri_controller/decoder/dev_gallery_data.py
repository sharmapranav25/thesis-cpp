"""Audited selection-only DEV font library; never a TRAIN reference reader."""
import collections
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .common import read,write,sha,now
from .data import pixel_inputs,resize_geometry,box_targets
from .dev_gallery import inputs as gallery_inputs,digest,initialize,one


def decoded(z,row,record):
    if record['split']!='dev' or record['role']!='selection_library':raise ValueError('DEV selection library required')
    rgb=np.asarray(Image.open(io.BytesIO(z.read(row['stem']+'.png'))).convert('RGB'))
    label=json.loads(z.read(row['stem']+'.json'));recipe=label['recipe']
    if (hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()!=row['pixel_sha256']
        or label['pixel_sha256']!=row['pixel_sha256'] or label['reference']!=record
        or not label['valid'] or label['count']!=6 or len(label['glyphs'])!=6 or label['rows']!=1
        or recipe['split']!='dev' or recipe['role']!='selection_library'
        or recipe['specimen_id']!=record['specimen_id'] or recipe['text']['text']!=record['text']
        or recipe['effects'] or any(x['present'] for x in label['effects'].values())
        or recipe['size']!=[rgb.shape[1],rgb.shape[0]]):raise ValueError('DEV library label/pixel identity mismatch')
    for i,glyph in enumerate(label['glyphs']):
        if (glyph['char']!=record['text'][i] or glyph['font_key']!=record['font_key']
            or glyph['font_sha256']!=record['font_sha256'] or glyph['axes']!=record['axes']):
            raise ValueError('DEV library glyph provenance mismatch')
        for key in ('box_core','box_visible'):
            b=glyph[key]
            if b is None or not (0<=b[0]<b[2]<=rgb.shape[1] and 0<=b[1]<b[3]<=rgb.shape[0]):
                raise ValueError('DEV library box bounds invalid')
    return rgb,label


def snapshot(run,decode_all=False):
    run=Path(run);root=run/'font_gallery_dev_v1';fonts,records,expected=gallery_inputs(run)
    protocol=read(root/'protocol.json');done=read(root/'completion.json');ph=digest(protocol)
    if protocol!=expected or done['state']!='complete' or done['protocol_hash']!=ph or done['split']!='dev':
        raise ValueError('Complete, unchanged DEV gallery required')
    archives={};rows=[];pixels=collections.Counter();families=collections.defaultdict(set)
    receipts=sorted(root.glob('shard_*.receipt.json'))
    if len(list(root.glob('shard_*.zip')))!=len(receipts):raise ValueError('Uncommitted DEV gallery archive')
    for shard,path in enumerate(receipts):
        receipt=read(path);archive=path.with_name(path.name.replace('.receipt.json','.zip'))
        if (path.name!=f'shard_{shard:05d}.receipt.json' or receipt['first']!=len(rows)
            or receipt['protocol_hash']!=ph or sha(archive)!=receipt['sha256']
            or archive.stat().st_size!=receipt['bytes'] or not receipt['records']):
            raise ValueError('DEV gallery archive identity/continuity mismatch')
        archives[str(archive)]=receipt['sha256']
        with zipfile.ZipFile(archive) as z:
            names=[r['stem']+suffix for r in receipt['records'] for suffix in ('.png','.json')]
            if len(set(names))!=len(names) or sorted(z.namelist())!=sorted(names) or (decode_all and z.testzip() is not None):
                raise ValueError('DEV gallery archive members/CRC mismatch')
            for r in receipt['records']:
                if len(rows)>=len(records):raise ValueError('Excess DEV records')
                record=records[len(rows)]
                if not r['accepted'] or record['specimen_id']!=r['specimen_id'] or r['stem']!=f'dev_reference_{len(rows):07d}':
                    raise ValueError('DEV gallery plan coverage mismatch')
                # Decode on every verification; this gallery is small. Labels are
                # not forwarded to the model or used to select query regions.
                rgb,label=decoded(z,r,record);height,width=rgb.shape[:2]
                rows.append(dict(archive=str(archive),stem=r['stem'],pixel_sha256=r['pixel_sha256'],record=record,
                    shape=resize_geometry(width,height)[2:]))
                pixels[r['pixel_sha256']]+=1;families[r['pixel_sha256']].add(record['family_group'])
    duplicates=sum(n-1 for n in pixels.values())
    if (len(rows)!=len(records) or done['accepted']!=len(records) or done['total']!=len(records)
        or done['shards']!=len(receipts) or done['duplicate_pixels']!=duplicates):raise ValueError('Incomplete DEV gallery')
    identity=dict(protocol_hash=ph,archives=archives,rows=rows)
    return dict(rows=rows,snapshot_hash=digest(identity),protocol_hash=ph,archive_hashes=archives,
        duplicate_pixels=duplicates,cross_family_pixel_groups=sum(len(v)>1 for v in families.values()),
        fonts=fonts,records=records)


def audit_gallery(run):
    state=snapshot(run,decode_all=True);initialize(state['fonts'])
    selected=sorted(set(np.linspace(0,len(state['rows'])-1,8,dtype=int)))
    for index in selected:
        row=state['rows'][index];replay=one(row['record'])
        if not replay['accepted'] or replay['pixel_sha256']!=row['pixel_sha256']:raise ValueError('DEV render replay mismatch')
        with zipfile.ZipFile(row['archive']) as z:
            if any(z.read(row['stem']+suffix)!=value for suffix,value in replay['payload'].items()):
                raise ValueError('DEV render annotation replay mismatch')
    result=dict(created_at=now(),passed=True,split='dev',role='selection_library',references=len(state['rows']),
        fonts=len(state['fonts']),families=len({r['family_group'] for r in state['records']}),
        snapshot_hash=state['snapshot_hash'],protocol_hash=state['protocol_hash'],archive_hashes=state['archive_hashes'],
        duplicate_pixels=state['duplicate_pixels'],cross_family_pixel_groups=state['cross_family_pixel_groups'],
        all_images_decoded=True,all_labels_verified=True,full_archive_hashes=True,render_replays=len(selected),
        source_sha256=sha(__file__),
        limitation='Library audit only. No trained embeddings/retrieval accuracy. Duplicates and ambiguity retained; TRAIN distractors still required in evaluation.')
    write(Path(run)/'dev_gallery_audit.json',result);return result


class DevGalleryDataset(Dataset):
    """Explicit DEV identity, compatible pixel/label envelope, no split spoofing."""
    def __init__(self,run):
        self.run=Path(run);audit=read(self.run/'dev_gallery_audit.json');state=snapshot(run)
        if not audit['passed'] or audit['snapshot_hash']!=state['snapshot_hash'] or audit['source_sha256']!=sha(__file__):
            raise ValueError('Matching DEV gallery audit required')
        self.rows=state['rows'];self.snapshot_hash=state['snapshot_hash'];self.handles=collections.OrderedDict()

    def __len__(self):return len(self.rows)

    def __getstate__(self):
        state=self.__dict__.copy();state['handles']=collections.OrderedDict();return state

    def close(self):
        for archive in self.handles.values():archive.close()
        self.handles.clear()

    def __getitem__(self,index):
        row=self.rows[index];path=row['archive']
        if path not in self.handles:
            self.handles[path]=zipfile.ZipFile(path)
            if len(self.handles)>4:self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(path);rgb,label=decoded(self.handles[path],row,row['record'])
        inputs,geometry=pixel_inputs(rgb)
        labels={k:row['record'][k] for k in ('specimen_id','font_key','font_sha256','family_group','style','split','axes','role')}
        labels['boxes_visible']=box_targets(label['glyphs'],'box_visible',geometry)
        return dict(inputs=inputs,library_labels=labels,audit=dict(stem=row['stem'],geometry=geometry,pixel_sha256=row['pixel_sha256']))


def check_reader(run):
    import torch
    from .reference_data import reference_batches,reference_collate
    torch.set_num_threads(2);dataset=DevGalleryDataset(run)
    try:
        batches=reference_batches(dataset)
        assert sorted(i for b in batches for i in b)==list(range(len(dataset)))
        chosen=sorted(set(np.linspace(0,len(dataset)-1,32,dtype=int)))
        for index in chosen:
            item=dataset[index];again=dataset[index]
            assert set(item['inputs'])=={'pixels','valid_pixels'}
            assert item['library_labels']['split']=='dev' and item['library_labels']['role']=='selection_library'
            assert item['library_labels']['boxes_visible'].shape==(6,4) and 'text' not in item['library_labels']
            assert torch.isfinite(item['inputs']['pixels']).all()
            assert all(torch.equal(item['inputs'][k],again['inputs'][k]) for k in item['inputs'])
        batch=reference_collate([dataset[i] for i in batches[0]])
        assert all(r['split']=='dev' for r in batch['library_labels'])
        row=dataset.rows[0]
        with zipfile.ZipFile(row['archive']) as z:
            for split in ('train','calibration','test'):
                try:decoded(z,row,dict(row['record'],split=split))
                except ValueError:pass
                else:raise AssertionError('Non-DEV specimen accepted')
            try:decoded(z,dict(row,pixel_sha256='0'*64),row['record'])
            except ValueError:pass
            else:raise AssertionError('Altered pixel identity accepted')
        result=dict(created_at=now(),passed=True,references=len(dataset),snapshot_hash=dataset.snapshot_hash,
            decoded_samples=len(chosen),all_samples_once=True,pixel_only_inputs=True,explicit_dev_identity=True,
            wrong_splits_rejected=True,altered_pixel_identity_rejected=True,
            source_sha256=sha(__file__),limitation='CPU reader checks only; no model training or retrieval performance.')
        write(Path(run)/'dev_gallery_reader_checks.json',result);return result
    finally:dataset.close()
