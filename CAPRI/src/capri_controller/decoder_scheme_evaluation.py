"""Fixed 100-image-per-scheme final-pilot diagnostic and procedural previews."""
import collections
import csv
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .decoder.common import ROOT,read,write,sha,now,StageLock
from .decoder import training
from .decoder.data import pixel_inputs,EFFECT_HEADS
from .decoder.font_extraction import visual_regions
from .decoder.timed_pilot_eval import gallery_rows,association
from .decoder.reference_data import TrainReferenceDataset,reference_collate
from .decoder.dev_gallery_data import DevGalleryDataset
from .decoder.metrics import count_statistics
from .decoder.checkpoints import require_finite
from .decoder_preview_render import POLICY,recipe,render_preview

RUN=ROOT/'runs/capri_decoder_500k_20261007'
PREVIOUS=RUN/'diagnostics/real_three_schemes_100_epoch8_20261008'
OUT=RUN/'diagnostics/real_three_schemes_100_epoch8_20261008_v2'
INPUT_PATHS=read(ROOT/'configs/local.json')['decoder_scheme_inputs']
CRYPT=Path(INPUT_PATHS['cryptbb'])
CITY=Path(INPUT_PATHS['city'])
SIMPLE=ROOT/INPUT_PATHS['simple']


def progress(stage,**fields):
    value=dict(updated_at=now(),stage=stage,**fields);write(OUT/'progress.json',value);print(json.dumps(value),flush=True)


def manifests():
    """Rank unique RGB hashes, without accessing labels or scoring information."""
    if (PREVIOUS/'image_manifests.json').exists():
        # Reuse the exact pre-inference selection from the numeric-roundoff failure.
        sets=read(PREVIOUS/'image_manifests.json')
        assert set(sets)=={'CryptBB','City','SimpleCaptcha'}
        assert all(len(rows)==100 and len({r['pixel_sha256'] for r in rows})==100 for rows in sets.values())
        progress('manifest_reused',images=300,manifest_sha256=sha(PREVIOUS/'image_manifests.json'))
        return sets
    sets={}
    for name,paths in [('CryptBB',CRYPT.glob('*.png')),('City',CITY.glob('*.png')),
                       ('SimpleCaptcha',(Path(r['source']) for r in read(SIMPLE/'inventory.json')['rows']))]:
        unique={}
        for path in paths:
            with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            unique.setdefault(digest,dict(source=str(path),pixel_sha256=digest))
        chosen=[]
        for digest in sorted(unique)[:100]:
            row=unique[digest];opaque=OUT/'inputs'/name/(digest+'.png');opaque.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(row['source']) as im:im.convert('RGB').save(opaque)
            chosen.append(dict(opaque_id=digest,image_path=str(opaque),**row))
        assert len(chosen)==100
        sets[name]=chosen
        progress('manifest',dataset=name,selected=100,unique_pool=len(unique))
    return sets


def build_gallery(model):
    refs=TrainReferenceDataset(RUN);devrefs=DevGalleryDataset(RUN)
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
            if processed%800<8:progress('library_gallery',images=processed,total=len(gallery))
    bank=torch.stack(vectors).cuda();ids=torch.tensor(ids,device='cuda')
    assert len(set(ids.cpu().tolist()))==len(families)
    write(OUT/'gallery.json',dict(specimens=processed,prototypes=len(bank),families=families))
    return bank,ids,families


def predict(model,row,bank,ids,families):
    rgb=np.asarray(Image.open(row['image_path']).convert('RGB'))
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixel_sha256']
    inputs,geometry=pixel_inputs(rgb)
    assert set(inputs)=={'pixels','valid_pixels'}
    pixels=inputs['pixels']
    if not torch.isfinite(pixels).all() or pixels.min() < -1e-6 or pixels.max() > 1+1e-6:
        raise ValueError('Preprocessing outside numerical rounding tolerance')
    inputs['pixels']=pixels.clamp(0,1)
    with torch.autocast('cuda',dtype=torch.float16):pred=model(**{k:v[None].cuda() for k,v in inputs.items()})
    require_finite(pred)
    logits=pred['count_logits'][0].float();prob=logits.softmax(-1)
    embeddings=torch.nn.functional.normalize(pred['font_embeddings'][0].float(),dim=-1)
    similarity=embeddings@bank.T
    scores=torch.full((len(similarity),len(families)),-torch.inf,device='cuda')
    scores.scatter_reduce_(1,ids[None].expand(len(similarity),-1),similarity,reduce='amax',include_self=True)
    ranked=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:5].cpu().tolist()
    nh,nw=geometry['native_hw'];h,w=geometry['valid_hw'];ph,pw=geometry['padded_hw']
    boxes=pred['boxes'][0].float().cpu()*torch.tensor([pw*nw/w,ph*nh/h,pw*nw/w,ph*nh/h])
    presence=pred['presence_logits'][0].float().sigmoid().cpu();visibility=pred['visibility_logits'][0].float().sigmoid().cpu()
    colours=pred['colour_rgb'][0].float().cpu().numpy()
    queries=[dict(query_id=q,box_xyxy=box.tolist(),presence=float(presence[q]),visibility=float(visibility[q]),
        colour_rgb=np.rint(colours[q]*255).clip(0,255).astype(int).tolist(),
        nearest_fonts=[dict(family=families[k],cosine=float(scores[q,k])) for k in ranked[q]]) for q,box in enumerate(boxes)]
    bg=pred['background_under'][0,:,:max(1,h//4),:max(1,w//4)].float().cpu().numpy()
    return dict(opaque_id=row['opaque_id'],pixel_sha256=row['pixel_sha256'],geometry=geometry,
        count=int(prob.argmax())+2,count_confidence=float(prob.max()),count_logits=logits.cpu().tolist(),
        count_probabilities=prob.cpu().tolist(),presence_logits=pred['presence_logits'][0].float().cpu().tolist(),
        selected_regions=len(visual_regions(pred,.5,.25)[0]['query_ids']),queries=queries,
        background_rgb=np.rint(np.median(bg,axis=(1,2))*255).clip(0,255).astype(int).tolist(),
        composition_probabilities=pred['composition_logits'][0].float().sigmoid().cpu().tolist(),
        row_probabilities=pred['row_logits'][0].float().softmax(-1).cpu().tolist(),
        effect_probabilities=dict(zip(EFFECT_HEADS,pred['effect_presence_logits'][0].float().sigmoid().cpu().tolist())),
        effect_intensity=dict(zip(EFFECT_HEADS,pred['intensity_predictions'][0].float().cpu().tolist())))


def render_panels(sets,records):
    fonts={f['key']:f for f in read(RUN/'fonts/font_manifest.json')['fonts']}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    all_reports={}
    for name,rows in sets.items():
        canvas=Image.new('RGB',(1260,4*180+55),'#f4f5f7');d=ImageDraw.Draw(canvas)
        for col,label in enumerate(['Real input','CAPRI + renderer: random text A','CAPRI + renderer: random text B']):d.text((12+col*420,12),label,fill='#17242e',font=font)
        reports=[]
        for i,(row,pred) in enumerate(zip(rows[:4],records[name][:4])):
            y=55+i*180;images=[Image.open(row['image_path']).convert('RGB')];variants=[]
            for v in range(2):
                seed=POLICY['seed']+i*2+v
                rec=recipe(pred,fonts,seed);stem=f'{name}_{i+1:02d}_{v+1}'
                write(OUT/'renders'/(stem+'.recipe.json'),rec)
                try:
                    rendered=render_preview(rec,fonts)
                    im=Image.fromarray(rendered['image']);im.save(OUT/'renders'/(stem+'.png'))
                    report=dict(valid=rendered['valid'],quality=rendered['quality'],glyphs=rendered['glyphs'],effect_log=rendered['effect_log'])
                    write(OUT/'renders'/(stem+'.audit.json'),report);variants.append(dict(file=stem+'.png',valid=report['valid'],quality=report['quality']))
                except Exception as e:
                    # Keep failed examples visibly failed; do not substitute or retune.
                    im=Image.new('RGB',(360,85),'#ffe4e4');ImageDraw.Draw(im).text((8,8),'Render failed: '+type(e).__name__,fill='black',font=font)
                    variants.append(dict(error=repr(e)));write(OUT/'renders'/(stem+'.error.json'),variants[-1])
                images.append(im)
            for col,im in enumerate(images):
                scale=min(396/im.width,100/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)))
                canvas.paste(im,(12+420*col,y+28))
            d.text((12,y),f'{name} #{i+1}: predicted count {pred["count"]}, confidence {pred["count_confidence"]:.1%}',fill='#17242e',font=font)
            d.text((432,y+135),'Unrefined preview; font and style unverified',fill='#53606a',font=font)
            reports.append(dict(opaque_id=pred['opaque_id'],variants=variants))
        canvas.save(OUT/(name+'_comparison.png'));all_reports[name]=reports
    write(OUT/'render_summary.json',all_reports)
    return all_reports


def score_cryptbb(sets,records):
    # Called only after all schemes' predictions and preview recipes are frozen.
    with (CRYPT/'labels.csv').open(encoding='utf-8-sig',newline='') as f:labels={r['filename']:r['label'] for r in csv.DictReader(f)}
    counts=[len(labels[Path(r['source']).name]) for r in sets['CryptBB']]
    assert all(2<=c<=8 for c in counts)
    stats=count_statistics([r['count_logits'] for r in records['CryptBB']],counts,[r['presence_logits'] for r in records['CryptBB']],.5)
    distribution=dict(collections.Counter(counts))
    return dict(count=stats,label_sha256=sha(CRYPT/'labels.csv'),true_count_distribution=distribution,
                posthoc_majority_count_baseline=max(distribution.values())/len(counts))


@torch.inference_mode()
def main():
    if OUT.exists():raise FileExistsError('Preserving existing run: '+str(OUT))
    import psutil
    own_tree={psutil.Process().pid,*[p.pid for p in psutil.Process().parents()]}
    for proc in psutil.process_iter(['cmdline']):
        if any(x in (proc.info['cmdline'] or []) for x in ['timed-pilot-worker','train-baseline','test-decoder-schemes']) and proc.pid not in own_tree:
            raise RuntimeError('Training or duplicate evaluation active')
    pilot=RUN/'diagnostics/timed_pilot_20261008';done=read(pilot/'completion.json')
    assert done['state']=='complete' and done['last']['epoch']==8
    checkpoint=Path(done['last']['checkpoint']);receipt=read(checkpoint.parent/'receipt.json');assert sha(checkpoint)==receipt['sha256']
    frozen=read(pilot/'protocol.json');assert all(sha(ROOT/p)==v for p,v in frozen['sources'].items())
    OUT.mkdir(parents=True)
    with StageLock(OUT,'evaluation'):
        protocol=dict(created_at=now(),checkpoint=str(checkpoint),checkpoint_sha256=receipt['sha256'],weights='ema',
            selection='100 smallest unique RGB SHA256 per dataset; no label-based selection; first4 for previews',
            input='RGB pixels and validity only; labels after every prediction and render is saved and frozen',
            thresholds=dict(presence=.5,visibility=.25),font_policy='Frozen pilot TRAIN+DEV gallery and max cosine family',
            preprocessing='Shared pixel_inputs; clamp interpolation rounding <=1e-6 to [0,1]; larger excursions fail',
            preview=POLICY,training=False,code_sha256=sha(__file__),preview_code_sha256=sha(ROOT/'src/capri_controller/decoder_preview_render.py'),
            limitation='Historical diagnostic, not untouched benchmark; no real font or box annotations; no answer recognition.')
        write(OUT/'protocol.json',protocol);sets=manifests();write(OUT/'image_manifests.json',sets)
        training.initialize();model=training.construct()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);require_finite(state['ema']);model.load_state_dict(state['ema'],strict=True);del state
        model.eval();model.requires_grad_(False);bank,ids,families=build_gallery(model)
        records={};started=time.time()
        for name,rows in sets.items():
            records[name]=[]
            for i,row in enumerate(rows):
                records[name].append(predict(model,row,bank,ids,families))
                if (i+1)%25==0:progress('prediction',dataset=name,images=i+1,total=100)
        write(OUT/'predictions.json',records)
        freeze=dict(created_at=now(),images=300,sha256=sha(OUT/'predictions.json'),seconds=time.time()-started)
        write(OUT/'prediction_freeze.json',freeze)
        del model,bank;torch.cuda.empty_cache()
        progress('rendering');previews=render_panels(sets,records)
        write(OUT/'render_freeze.json',{p.name:sha(p) for p in sorted((OUT/'renders').glob('*'))})
        progress('scoring');crypt_score=score_cryptbb(sets,records)
        assert sha(OUT/'predictions.json')==freeze['sha256']
        summary=dict(completed_at=now(),datasets={name:dict(images=100,predicted_count_distribution=dict(collections.Counter(r['count'] for r in rows)),
            mean_selected_regions=float(np.mean([r['selected_regions'] for r in rows])),count_accuracy=None,font_accuracy=None,localization_accuracy=None) for name,rows in records.items()},
            CryptBB_scoring=crypt_score,limitation=protocol['limitation'],preview_policy=POLICY,training_started=False)
        summary['datasets']['CryptBB']['count_accuracy']=crypt_score['count']['accuracy']
        write(OUT/'summary.json',summary)
        page='<html><meta charset="utf-8"><title>CAPRI 100-image scheme tests</title><style>body{font:17px system-ui;max-width:1260px;margin:25px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>CAPRI: 100 images per dataset</h1><p>Final epoch8 EMA. Left: real pixels. Right: fresh random text rendered from predicted count, boxes, fonts, colours and effects. No controllers, training, target strings or VLMs. Font retrieval remains weak; previews are approximate, not validated authentic replicas. First four hash-ranked inputs, two fixed seeds; no selection by outcome.</p>'
        for name in records:page+='<h2>'+html.escape(name)+'</h2><img src="'+name+'_comparison.png">'
        page+='<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre></html>';(OUT/'report.html').write_text(page,encoding='utf8')
        progress('complete',images=300);print(json.dumps(summary,indent=2),flush=True)
