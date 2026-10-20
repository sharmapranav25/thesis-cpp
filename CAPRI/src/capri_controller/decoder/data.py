"""Pixel-only preprocessing and supervised reads of immutable decoder corpus shards.

No model is loaded and no training is started here. Targets are deliberately
separated from the two tensor fields that may be passed to the visual model.
"""
import collections
import hashlib
import io
import json
import math
import random
import struct
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .common import read, sha
from .appearance import font_targets

EFFECT_HEADS = tuple(sorted(('strike_through','bands','glyph_fill_texture','scribbles','hollow_outline_glyphs',
    'noise_over_text','low_resolution','straight_lines','glyph_cuts','perspective','blur','bg_gradient',
    'inverted_regions','blob_noise','bg_pattern','translucent_text','extrude_3d','colour_quantization',
    'stroke_bold','local_warp','bg_grid','bg_image_texture','stroke_thin','glyph_outline','bg_circles',
    'morph_distortion','dot_noise','occlusion','pixel_noise','bg_decoy_chars','bg_blocks','glyph_shadow',
    'pixelation','border_frame','curves','compression','wave_warp')))


def resize_geometry(width,height):
    if width<=0 or height<=0:raise ValueError('Invalid image dimensions')
    scale=min(128/height,768/width)
    valid_h=max(1,round(height*scale)); valid_w=max(1,round(width*scale))
    return valid_h,valid_w,math.ceil(valid_h/16)*16,math.ceil(valid_w/16)*16


def pixel_inputs(rgb):
    """Shared train/inference transform; accepts only RGB pixels, never a label."""
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('Expected uint8 HxWx3 RGB')
    height,width=rgb.shape[:2]
    h,w,ph,pw=resize_geometry(width,height)
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().div_(255)
    x=F.interpolate(x[None],size=(h,w),mode='bilinear',align_corners=False,antialias=True)[0]
    x=F.pad(x,(0,pw-w,0,ph-h),value=1.)
    valid=torch.zeros(ph,pw,dtype=torch.bool);valid[:h,:w]=True
    return dict(pixels=x,valid_pixels=valid),dict(native_hw=(height,width),valid_hw=(h,w),padded_hw=(ph,pw))


def coverage_target(array,geometry,stride=4):
    h,w=geometry['valid_hw'];ph,pw=geometry['padded_hw']
    tensor=torch.from_numpy(np.asarray(array).copy()).float().div_(255)
    tensor=F.interpolate(tensor[None],size=(h,w),mode='bilinear',align_corners=False,antialias=True)[0]
    tensor=F.pad(tensor,(0,pw-w,0,ph-h))
    return F.interpolate(tensor[None],size=(ph//stride,pw//stride),mode='area')[0]


def box_targets(glyphs,field,geometry):
    nh,nw=geometry['native_hw'];h,w=geometry['valid_hw'];ph,pw=geometry['padded_hw']
    boxes=torch.tensor([g[field] for g in glyphs],dtype=torch.float32)
    return boxes*torch.tensor([w/nw/pw,h/nh/ph,w/nw/pw,h/nh/ph])


class CorpusDataset(Dataset):
    """Take a fixed snapshot of complete, verified shards in one declared split.

    allow_partial is for diagnostic I/O checks while generation runs. Production
    training requires the completion receipt; a live dataset never grows itself.
    """
    def __init__(self,run,split='train',allow_partial=False,max_shards=None,purpose='optimization'):
        self.run=Path(run);self.split=split
        expected={'train':'optimization','dev':'selection','calibration':'calibration','test':'final_score'}
        if split not in expected or purpose!=expected[split]:raise ValueError('Split/purpose mismatch')
        root=self.run/'corpus'
        self.protocol=read(self.run/'corpus_protocol.json')
        self.protocol_hash=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()
        if not allow_partial:
            done=read(root/'completion.json')
            if done['state']!='complete' or done['protocol_hash']!=self.protocol_hash or done['accepted_unique']!=500000:
                raise ValueError('Completed corpus required before training/evaluation')
        if split in ('calibration','test'):
            selection=read(self.run/'selection_frozen.json')
            if selection.get('corpus_protocol_hash')!=self.protocol_hash:
                raise ValueError('Freeze checkpoint selection before calibration/test')
            if sha(selection['checkpoint_path'])!=selection['checkpoint_sha256']:
                raise ValueError('Selected checkpoint hash mismatch')
            if split=='test':
                calibration=read(self.run/'calibration_frozen.json')
                if calibration.get('checkpoint_sha256')!=selection['checkpoint_sha256']:
                    raise ValueError('Freeze matching calibration before final test')
        manifest=read(self.run/'fonts/font_manifest.json')
        if sha(self.run/'fonts/font_manifest.json')!=self.protocol['font_manifest_sha256']:
            raise ValueError('Font manifest differs from frozen corpus')
        self.fonts={f['key']:f for f in manifest['fonts']}
        families=sorted({f['split_group'] for f in self.fonts.values()})
        self.family_indices={name:i for i,name in enumerate(families)}
        self.rows=[];self.handles=collections.OrderedDict()
        receipts=sorted((root/split).glob('*.receipt.json'))
        if max_shards is not None:
            if not allow_partial:raise ValueError('Subset reads are diagnostic only')
            receipts=receipts[:max_shards]
        for receipt_path in receipts:
            receipt=read(receipt_path)
            archive=receipt_path.with_name(receipt_path.name.replace('.receipt.json','.zip'))
            if receipt['split']!=split or receipt['protocol_hash']!=self.protocol_hash or sha(archive)!=receipt['sha256']:
                raise ValueError('Shard identity/integrity mismatch')
            with zipfile.ZipFile(archive) as z:
                for record in receipt['images']:
                    stem=record['stem']
                    with z.open(stem+'.png') as f:header=f.read(24)
                    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise ValueError('Invalid PNG header')
                    width,height=struct.unpack('>II',header[16:24])
                    self.rows.append(dict(archive=str(archive),stem=stem,pixel_sha256=record['pixel_sha256'],
                                          native_hw=(height,width),shape=resize_geometry(width,height)[2:]))
        if not self.rows:raise ValueError('No complete shards in split '+split)
        if not allow_partial and len(self.rows)!=self.protocol['split_sizes'][split]:
            raise ValueError('Incomplete split')
        self.snapshot_hash=hashlib.sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()

    def __len__(self):return len(self.rows)

    def __getstate__(self):
        state=self.__dict__.copy();state['handles']=collections.OrderedDict();return state

    def close(self):
        for handle in self.handles.values():handle.close()
        self.handles.clear()

    def archive(self,path):
        if path in self.handles:self.handles.move_to_end(path)
        else:
            self.handles[path]=zipfile.ZipFile(path)
            if len(self.handles)>4:self.handles.popitem(last=False)[1].close()
        return self.handles[path]

    def __getitem__(self,index):
        row=self.rows[index];z=self.archive(row['archive']);stem=row['stem']
        rgb=np.asarray(Image.open(io.BytesIO(z.read(stem+'.png'))).convert('RGB'))
        if hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()!=row['pixel_sha256']:
            raise ValueError('Pixel checksum mismatch')
        inputs,geometry=pixel_inputs(rgb)
        label=json.loads(z.read(stem+'.json'));recipe=label['recipe'];glyphs=label['glyphs']
        if recipe['split']!=self.split or set(label['effects'])!=set(EFFECT_HEADS):raise ValueError('Label schema/split mismatch')
        with np.load(io.BytesIO(z.read(stem+'.masks.npz')),allow_pickle=False) as masks:
            instances={name:coverage_target(masks[name],geometry) for name in ('visible','geometry','decoration')}
        if any(m.shape[0]!=label['count'] for m in instances.values()):raise ValueError('Instance count mismatch')
        fonts=[self.fonts[g['font_key']] for g in glyphs]
        if any(f['split']!=self.split for f in fonts):raise ValueError('Cross-split font label')
        presence=torch.tensor([label['effects'][head]['present'] for head in EFFECT_HEADS],dtype=torch.float32)
        level=torch.tensor([label['effects'][head]['level'] for head in EFFECT_HEADS],dtype=torch.long)
        valid=F.avg_pool2d(inputs['valid_pixels'][None,None].float(),4)[0,0]
        background=np.asarray(Image.open(io.BytesIO(z.read(stem+'.background.png'))).convert('RGB'))
        background=coverage_target(background.transpose(2,0,1),geometry)
        # Strings are reduced to composition supervision, never forwarded as input.
        chars=[g['char'] for g in glyphs]
        targets=dict(count=torch.tensor(label['count'],dtype=torch.long),
                     count_class=torch.tensor(label['count']-2,dtype=torch.long),
                     boxes_visible=box_targets(glyphs,'box_visible',geometry),
                     boxes_core=box_targets(glyphs,'box_core',geometry),masks=instances,valid_mask=valid,
                     visibility=torch.tensor([g['visible_fraction'] for g in glyphs],dtype=torch.float32),
                     font_family=torch.tensor([self.family_indices[f['split_group']] for f in fonts],dtype=torch.long),
                     effect_presence=presence,effect_level=level,effect_intensity=level.float()/10,
                     intensity_valid=presence.bool(),
                     composition=torch.tensor([any(c.isdigit() for c in chars),any(c.isupper() for c in chars),
                                               any(c.islower() for c in chars)],dtype=torch.float32),
                     rows=torch.tensor(label['rows'],dtype=torch.long),background_under=background,
                     colour_rgb=torch.tensor([g['final_colour_median'] if g['final_colour_median'] is not None else [float('nan')]*3 for g in glyphs],dtype=torch.float32)/255,
                     font_attributes=font_targets(glyphs,fonts),
                     glyph_metadata=[{key:g.get(key) for key in ('angle','shear','stretch','axes','italic','category',
                                      'stroke_class','x_height','intrinsic_width','stroke_width')} for g in glyphs])
        audit=dict(stem=stem,geometry=geometry,pair_group=recipe['pair_group'],pair_kind=recipe['pair_kind'],
                   font_keys=[g['font_key'] for g in glyphs],quality=label['quality'])
        return dict(inputs=inputs,targets=targets,audit=audit)


def exact_shape_collate(items):
    shapes={tuple(item['inputs']['pixels'].shape) for item in items}
    if len(shapes)!=1:raise ValueError('Mixed shapes would introduce batch-dependent DINO padding')
    return dict(inputs={key:torch.stack([item['inputs'][key] for item in items]) for key in ('pixels','valid_pixels')},
                targets=[item['targets'] for item in items],audit=[item['audit'] for item in items])


class ShapeBatchPlan:
    """Reproducible exact-shape batches retaining declared two-image font pairs.

    State is the immutable snapshot hash, epoch, seed and CONSUMED batch cursor.
    The trainer owns that cursor: DataLoader prefetch never advances resume state.
    """
    def __init__(self,dataset,batch_size=4,epoch=0,seed=731,consumed_batches=0,shuffle=True):
        if batch_size<2:raise ValueError('Batch size must allow a paired unit')
        buckets=collections.defaultdict(list);seen=set()
        lookup={row['stem']:i for i,row in enumerate(dataset.rows)}
        for index,row in enumerate(dataset.rows):
            if index in seen:continue
            number=int(row['stem'].rsplit('_',1)[1]);unit=[index]
            if (number//2)%10<3:
                mate=row['stem'].rsplit('_',1)[0]+'_'+f'{number^1:07d}'
                other=lookup.get(mate)
                if other is None or other in seen or dataset.rows[other]['shape']!=row['shape']:
                    raise ValueError('Missing/incompatible paired sample')
                unit.append(other)
            seen.update(unit);buckets[tuple(row['shape'])].append(unit)
        rng=random.Random(seed+1000003*epoch);batches=[]
        for shape,units in sorted(buckets.items()):
            if shuffle:rng.shuffle(units)
            batch=[]
            for unit in units:
                if len(batch)+len(unit)>batch_size:batches.append(batch);batch=[]
                batch.extend(unit)
            if batch:batches.append(batch)
        if shuffle:rng.shuffle(batches)
        if not 0<=consumed_batches<=len(batches):raise ValueError('Invalid consumed batch cursor')
        self.batches=batches;self.consumed_batches=consumed_batches
        self.state=dict(snapshot_hash=dataset.snapshot_hash,batch_size=batch_size,epoch=epoch,seed=seed,
                        consumed_batches=consumed_batches,shuffle=shuffle)

    def __iter__(self):return iter(self.batches[self.consumed_batches:])
    def __len__(self):return len(self.batches)-self.consumed_batches

    @classmethod
    def from_state(cls,dataset,state):
        if state['snapshot_hash']!=dataset.snapshot_hash:
            raise ValueError('Sampler resume dataset snapshot mismatch')
        return cls(dataset,**{k:state[k] for k in ('batch_size','epoch','seed','consumed_batches','shuffle')})
