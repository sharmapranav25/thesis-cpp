"""I/O and sampling tests on existing train shards only; no optimizer or model."""
import copy
import io
import json
import time
import numpy as np
from PIL import Image
import torch
from .common import write,now,sha
from .data import CorpusDataset,ShapeBatchPlan,exact_shape_collate,pixel_inputs,EFFECT_HEADS


def check_data(run):
    torch.set_num_threads(2)
    start=time.monotonic();ds=CorpusDataset(run,allow_partial=True,max_shards=1)
    plan=ShapeBatchPlan(ds,batch_size=4,epoch=2)
    batches=list(plan);indices=[i for batch in batches for i in batch]
    assert sorted(indices)==list(range(len(ds)))
    assert len(EFFECT_HEADS)==37 and len(set(EFFECT_HEADS))==37
    for batch in batches:
        assert len({ds.rows[i]['shape'] for i in batch})==1
        assert len(batch)<=4
    positions={i:b for b,batch in enumerate(batches) for i in batch}
    for i,row in enumerate(ds.rows):
        number=int(row['stem'].rsplit('_',1)[1])
        if (number//2)%10<3:
            assert positions[i]==positions[i+1 if number%2==0 else i-1]
    assert batches==list(ShapeBatchPlan(ds,batch_size=4,epoch=2))
    assert batches!=list(ShapeBatchPlan(ds,batch_size=4,epoch=3))
    state=copy.deepcopy(plan.state);state['consumed_batches']=17
    assert list(ShapeBatchPlan.from_state(ds,state))==batches[17:]
    bad=copy.deepcopy(state);bad['snapshot_hash']='changed'
    try:ShapeBatchPlan.from_state(ds,bad)
    except ValueError:pass
    else:raise AssertionError('Changed sampler snapshot accepted')
    checked=0;seen_shapes={};count_support=set()
    # Deterministic spread plus first seven paired count cases.
    chosen=sorted(set(range(14))|set(np.linspace(0,len(ds)-1,40,dtype=int).tolist()))
    for index in chosen:
        item=ds[index];x=item['inputs'];target=item['targets'];geometry=item['audit']['geometry']
        assert set(x)=={'pixels','valid_pixels'}
        assert x['pixels'].dtype==torch.float32 and x['valid_pixels'].dtype==torch.bool
        assert torch.isfinite(x['pixels']).all() and x['pixels'].min()>=0 and x['pixels'].max()<=1
        ph,pw=geometry['padded_hw'];h,w=geometry['valid_hw']
        assert x['pixels'].shape==(3,ph,pw) and int(x['valid_pixels'].sum())==h*w
        assert target['masks']['visible'].shape==(int(target['count']),ph//4,pw//4)
        assert ((target['boxes_visible']>=0)&(target['boxes_visible']<=1)).all()
        assert (target['boxes_visible'][:,2:]>target['boxes_visible'][:,:2]).all()
        assert torch.equal(target['intensity_valid'],target['effect_presence'].bool())
        assert ((target['effect_level']>=0)&(target['effect_level']<=10)).all()
        assert int(target['count_class'])==int(target['count'])-2
        count_support.add(int(target['count']))
        row=ds.rows[index];z=ds.archive(row['archive'])
        rgb=np.asarray(Image.open(io.BytesIO(z.read(row['stem']+'.png'))).convert('RGB'))
        direct,_=pixel_inputs(rgb)
        assert all(torch.equal(direct[key],x[key]) for key in x)
        raw=json.loads(z.read(row['stem']+'.json'))
        # Undo the normalized visible-box transform and recover original coordinates.
        nh,nw=geometry['native_hw']
        recovered=target['boxes_visible']*torch.tensor([pw*nw/w,ph*nh/h,pw*nw/w,ph*nh/h])
        assert torch.allclose(recovered,torch.tensor([g['box_visible'] for g in raw['glyphs']],dtype=torch.float32),atol=1e-4)
        seen_shapes.setdefault((ph,pw),item);checked+=1
    assert count_support==set(range(2,9))
    a,b=list(seen_shapes.values())[:2]
    try:exact_shape_collate([a,b])
    except ValueError:pass
    else:raise AssertionError('Batch-dependent padding silently introduced')
    batch=exact_shape_collate([a,a])
    assert torch.equal(batch['inputs']['pixels'][0],batch['inputs']['pixels'][1])
    try:CorpusDataset(run,split='test',allow_partial=True,purpose='optimization')
    except ValueError:pass
    else:raise AssertionError('Final test split accepted for optimization')
    # Windows spawn pickling must never serialize open ZIP handles.
    assert ds.handles and not ds.__getstate__()['handles']
    ds.close()
    result=dict(created_at=now(),passed=True,train_rows_indexed=len(ds),examples_checked=checked,
                exact_shape_batches=len(batches),source_sha256=sha(__import__('pathlib').Path(__file__).with_name('data.py')),
                snapshot_hash=ds.snapshot_hash,seconds=time.monotonic()-start,
                checks=['pixel-only input contract','train/inference preprocessing equality','count and box transforms',
                        'mask dimensions and conditional intensity validity','all examples once per epoch',
                        'font/content pairs retained in same shape batch','deterministic consumed-cursor resume',
                        'snapshot mismatch refusal','mixed-shape batch refusal','test-for-optimization refusal','spawn-safe ZIP state'],
                limitation='Diagnostic reads from one completed TRAIN shard while generation runs. No final-test scoring or model/optimizer executed; production training still requires corpus completion and pretrained weights.')
    write(run/'data_checks.json',result)
    return result
