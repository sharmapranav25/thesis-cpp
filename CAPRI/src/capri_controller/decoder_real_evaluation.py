"""Frozen final-pilot real-image diagnostic; labels are accessed only after prediction freeze."""
import collections
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .decoder.common import ROOT, read, write, sha, now, StageLock
from .decoder import training
from .decoder.data import pixel_inputs, EFFECT_HEADS
from .decoder.font_extraction import visual_regions
from .decoder.timed_pilot_eval import gallery_rows, association
from .decoder.reference_data import TrainReferenceDataset, reference_collate
from .decoder.dev_gallery_data import DevGalleryDataset
from .decoder.metrics import count_statistics
from .decoder.checkpoints import require_finite


@torch.inference_mode()
def main():
    run=ROOT/'runs/capri_decoder_500k_20261007'
    pilot=run/'diagnostics/timed_pilot_20261008'
    out=run/'diagnostics/real_darkbay_epoch8_20261008'
    if out.exists():raise FileExistsError('Preserving existing real evaluation: '+str(out))
    import psutil
    for proc in psutil.process_iter(['cmdline']):
        if any(x in (proc.info['cmdline'] or []) for x in ['timed-pilot-worker','train-baseline']):
            raise RuntimeError('Training still active')
    done=read(pilot/'completion.json');assert done['state']=='complete' and done['last']['epoch']==8
    checkpoint=Path(done['last']['checkpoint']);receipt=read(checkpoint.parent/'receipt.json')
    assert sha(checkpoint)==receipt['sha256']
    frozen=read(pilot/'protocol.json')
    assert all(sha(ROOT/path)==digest for path,digest in frozen['sources'].items())
    source=ROOT/'runs/parseq_darkbay_100k_20261005/evaluation_darkbay'
    manifest=read(source/'image_manifest.json')
    assert len(manifest)==900 and len({r['opaque_id'] for r in manifest})==900
    out.mkdir(parents=True)
    with StageLock(out,'evaluation'):
        protocol=dict(created_at=now(),checkpoint=str(checkpoint),checkpoint_sha256=receipt['sha256'],weights='ema',
            selection='Final epoch8 fixed before real labels; no checkpoint sweep',manifest_sha256=sha(source/'image_manifest.json'),
            images=len(manifest),instance_threshold=.5,visibility_threshold=.25,
            preprocessing='Shared training pixel_inputs; RGB aspect-preserving resize and white padding',
            font_policy='Same TRAIN+DEV first3 specimens per font file and max cosine per family as pilot; scores are not probabilities',
            input='pixels and validity mask only',label_access='After complete predictions and SHA256 freeze; lengths only for scoring',
            limitation='Historical repeatedly examined DarkBay set; count scoring only, no real box/font annotations or text-recognition head',
            code_sha256=sha(__file__),frozen_sources=frozen['sources'])
        write(out/'protocol.json',protocol);write(out/'image_manifest.json',manifest)
        training.initialize();model=training.construct()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);require_finite(state['ema'])
        model.load_state_dict(state['ema'],strict=True);del state
        model.eval();model.requires_grad_(False)
        refs=TrainReferenceDataset(run);devrefs=DevGalleryDataset(run)
        gallery=gallery_rows(refs,devrefs);buckets=collections.defaultdict(list)
        for ds,i in gallery:buckets[tuple(ds.rows[i]['shape'])].append((ds,i))
        families=sorted({ds.rows[i]['record']['family_group'] for ds,i in gallery})
        family_id={f:i for i,f in enumerate(families)};vectors=[];ids=[];processed=0
        for items in buckets.values():
            for start in range(0,len(items),8):
                batch=reference_collate([ds[i] for ds,i in items[start:start+8]])
                with torch.autocast('cuda',dtype=torch.float16):pred=model(**{k:v.cuda() for k,v in batch['inputs'].items()})
                for region,label in zip(visual_regions(pred,.5,.25),batch['library_labels']):
                    for pi,ti in association(region['boxes'],label['boxes_visible']):
                        vectors.append(region['embeddings'][pi].cpu());ids.append(family_id[label['family_group']])
                processed+=len(batch['library_labels'])
                if processed%400<8:print(json.dumps(dict(stage='gallery',images=processed,total=len(gallery))),flush=True)
        bank=torch.stack(vectors).cuda();ids=torch.tensor(ids,device='cuda');del vectors
        assert len(set(ids.cpu().tolist()))==len(families)
        write(out/'gallery.json',dict(specimens=processed,prototypes=len(bank),families=families))
        records=[];started=time.time()
        # All paths are opaque copies; no answer-bearing source filenames enter inference.
        for index,row in enumerate(manifest):
            rgb=np.asarray(Image.open(row['image_path']).convert('RGB'))
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixel_sha256']
            inputs,geometry=pixel_inputs(rgb)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(**{k:v[None].cuda() for k,v in inputs.items()})
            require_finite(pred)
            logits=pred['count_logits'][0].float();prob=logits.softmax(-1)
            region=visual_regions(pred,.5,.25)[0];similarity=region['embeddings']@bank.T
            scores=torch.full((len(similarity),len(families)),-torch.inf,device='cuda')
            scores.scatter_reduce_(1,ids[None].expand(len(similarity),-1),similarity,reduce='amax',include_self=True)
            ranked=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:5].cpu().tolist()
            nh,nw=geometry['native_hw'];h,w=geometry['valid_hw'];ph,pw=geometry['padded_hw']
            native=region['boxes'].cpu()*torch.tensor([pw*nw/w,ph*nh/h,pw*nw/w,ph*nh/h])
            glyphs=[dict(query_id=int(q),box_xyxy=box.tolist(),quality=float(region['quality'][j]),
                nearest_fonts=[dict(family=families[k],cosine=float(scores[j,k])) for k in ranked[j]])
                for j,(q,box) in enumerate(zip(region['query_ids'],native))]
            effects=pred['effect_presence_logits'][0].float().sigmoid().cpu().tolist()
            records.append(dict(opaque_id=row['opaque_id'],pixel_sha256=row['pixel_sha256'],
                count=int(prob.argmax())+2,count_confidence=float(prob.max()),count_logits=logits.cpu().tolist(),
                count_probabilities=prob.cpu().tolist(),presence_logits=pred['presence_logits'][0].float().cpu().tolist(),
                selected_regions=len(glyphs),glyphs=glyphs,geometry=geometry,
                effect_probabilities=dict(zip(EFFECT_HEADS,effects)),
                effect_intensity=dict(zip(EFFECT_HEADS,pred['intensity_predictions'][0].float().cpu().tolist()))))
            if (index+1)%100==0:print(json.dumps(dict(stage='real_predictions',images=index+1,total=len(manifest))),flush=True)
        write(out/'predictions.json',records)
        freeze=dict(created_at=now(),images=len(records),sha256=sha(out/'predictions.json'),checkpoint_sha256=receipt['sha256'],seconds=time.time()-started)
        write(out/'prediction_freeze.json',freeze)
        # Scoring begins here, after every prediction is persisted and hash-frozen.
        labels=read(source/'scoring_labels.json')
        assert set(labels)=={r['opaque_id'] for r in records}
        counts=[len(labels[r['opaque_id']]) for r in records];del labels
        assert sha(out/'predictions.json')==freeze['sha256']
        stats=count_statistics([r['count_logits'] for r in records],counts,[r['presence_logits'] for r in records],.5)
        summary=dict(completed_at=now(),images=len(records),checkpoint_sha256=receipt['sha256'],count=stats,
            predicted_count_distribution=dict(collections.Counter(r['count'] for r in records)),
            true_count_distribution=dict(collections.Counter(counts)),
            font_accuracy=None,localization_accuracy=None,text_recognition_accuracy=None,
            limitation=protocol['limitation'],training_started=False,real_threshold_tuning=False)
        write(out/'summary.json',summary)
        examples(out,manifest,records,counts)
        page='<html><meta charset="utf-8"><title>CAPRI real DarkBay diagnostic</title><style>body{font:17px system-ui;max-width:1100px;margin:32px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>CAPRI final pilot: real DarkBay</h1><p>Count accuracy only. Boxes and nearest-font candidates are unscored predictions. No reader decoding, training or threshold tuning.</p><img src="examples.png"><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre></html>'
        (out/'report.html').write_text(page,encoding='utf8')
        print(json.dumps(summary,indent=2),flush=True)


def examples(out,manifest,records,counts):
    """First twelve fixed manifest entries; never choose examples by outcome."""
    canvas=Image.new('RGB',(1100,6*190),'#f5f6f8');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for i,(row,pred,truth) in enumerate(zip(manifest[:12],records[:12],counts[:12])):
        x=(i%2)*550;y=(i//2)*190
        image=Image.open(row['image_path']).convert('RGB');scale=min(510/image.width,110/image.height)
        image=image.resize((round(image.width*scale),round(image.height*scale)))
        canvas.paste(image,(x+15,y+40))
        for glyph in pred['glyphs']:
            b=glyph['box_xyxy'];draw.rectangle((x+15+b[0]*scale,y+40+b[1]*scale,x+15+b[2]*scale,y+40+b[3]*scale),outline='#d12c38',width=2)
        draw.text((x+15,y+8),f"#{i+1} Count {pred['count']} | true {truth} | confidence {pred['count_confidence']:.1%}",fill='#222222',font=font)
        draw.text((x+15,y+157),f"Detected regions: {pred['selected_regions']} (boxes unscored)",fill='#444444',font=font)
    canvas.save(out/'examples.png')
