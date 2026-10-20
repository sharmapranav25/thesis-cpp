"""Freeze three final-checkpoint predictions before scoring real DarkBay labels."""
from pathlib import Path
import collections,copy,gc,hashlib,html,json,re,sys,time,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .. import runtime as U
from .prepare_parseq import RUN,OPENOCR

SOURCE=Path('E:/pranav/DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled')
OUT=RUN/'evaluation_darkbay'

def main():
    if OUT.exists():raise FileExistsError('Evaluation already exists; preserve the frozen readout: '+str(OUT))
    OUT.mkdir();(OUT/'images').mkdir()
    completion=U.read(RUN/'completion.json');assert completion['all_complete']
    for run in completion['seeds']:
        assert run['epoch']==20 and U.sha(run['checkpoint'])==run['sha256']
    excluded={r['pixel_sha256'] for r in U.read(U.WORK/'capri_darkbay_workflow/inputs.json')}
    by_hash={};excluded_found=set();sources=0;sizes=collections.Counter()
    for path in sorted(SOURCE.glob('*.png')):
        image=Image.open(path).convert('RGB');digest=hashlib.sha256(np.asarray(image).tobytes()).hexdigest();sources+=1;sizes[str(image.size)]+=1
        if digest in excluded:excluded_found.add(digest);continue
        by_hash.setdefault(digest,[]).append(str(path))
    assert excluded_found==excluded,'Could not locate all100 controller inputs by RGB hash'
    records=[];provenance={}
    for digest,paths in sorted(by_hash.items()):
        opaque_id='darkbay_'+digest;path=OUT/'images'/(opaque_id+'.png')
        Image.open(paths[0]).convert('RGB').save(path)
        records.append(dict(opaque_id=opaque_id,image_path=str(path),pixel_sha256=digest));provenance[opaque_id]=paths
    # Prove no exact rendered-pixel overlap with the actual synthetic corpus.
    synthetic=set()
    for path in sorted((U.WORK/'capri_darkbay_workflow/corpus').glob('*.zip')):
        with zipfile.ZipFile(path) as z:synthetic.update(json.loads(line)['pixel_sha256'] for line in z.read('labels.jsonl').decode().splitlines())
    assert len(synthetic)==100000 and not synthetic.intersection(by_hash)
    U.dump(OUT/'image_manifest.json',records);U.dump(OUT/'source_provenance.json',provenance)
    U.dump(OUT/'protocol.json',dict(source_images=sources,evaluation_unique_images=len(records),excluded_controller_images=len(excluded_found),
        duplicate_eligible_images=sum(len(v)-1 for v in by_hash.values()),synthetic_pixel_overlap=0,source_sizes=dict(sizes),
        metric='Case-sensitive exact whole-string match; corpus CER also reported',checkpoint_selection='All three fixed epoch20 latest checkpoints, no winner selected',
        preprocessing='Training-compatible RGB torchvision bicubic32x128, ToTensor, Normalize(.5,.5)',decoding='Saved PARSeq autoregressive decoder with one refinement; no forced length or charset narrowing',
        label_access='Ground-truth filename stems are parsed only after all three prediction files are hash-frozen',
        limitation='Existing historical real-family images; excludes this run\'s100 controller references but does not establish untouched status relative to earlier experiments.',
        checkpoints=completion['seeds'],code_sha256=U.sha(Path(__file__))))
    print('Evaluation images:',len(records),'Excluded controller inputs:',len(excluded_found),flush=True)
    sys.path[:0]=[str(U.WORK/'.runtime/parseq_packages'),str(OPENOCR)]
    import torch
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    torch.set_num_threads(8);device='cuda' if torch.cuda.is_available() else 'cpu'
    transform=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    predictions={};frozen={}
    for run in completion['seeds']:
        seed=run['seed'];checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=False);cfg=copy.deepcopy(checkpoint['config'])
        assert cfg['Global']['use_space_char'] is False
        decoder=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.to(device).eval();del checkpoint
        rows=[];started=time.time()
        with torch.inference_mode():
            for start in range(0,len(records),64):
                chunk=records[start:start+64];x=torch.stack([transform(Image.open(r['image_path']).convert('RGB')) for r in chunk]).to(device)
                outputs=decoder(model(x))
                assert len(outputs)==len(chunk)
                for record,(text,confidence) in zip(chunk,outputs):
                    rows.append(dict(opaque_id=record['opaque_id'],prediction=text,confidence=float(confidence) if np.isfinite(confidence) else None))
        path=OUT/f'seed_{seed}_predictions.json';U.dump(path,rows);frozen[str(seed)]=dict(file=path.name,sha256=U.sha(path),count=len(rows),seconds=time.time()-started)
        predictions[seed]={r['opaque_id']:r['prediction'] for r in rows};del model;gc.collect();torch.cuda.empty_cache()
        print('Frozen predictions for seed',seed,flush=True)
    U.dump(OUT/'prediction_freeze.json',frozen)
    # Scoring starts only after every final prediction is immutable and hashed.
    from rapidfuzz.distance import Levenshtein
    labels={}
    for record in records:
        candidates={Path(path).stem for path in provenance[record['opaque_id']]}
        if len(candidates)!=1:raise ValueError('Conflicting labels for duplicate image: '+record['opaque_id'])
        text=candidates.pop()
        if not re.fullmatch('[0-9A-Za-z]+',text):raise ValueError('Unexpected label format: '+text)
        labels[record['opaque_id']]=text
    U.dump(OUT/'scoring_labels.json',labels)
    results=[];details=[]
    for seed,pred in predictions.items():
        assert U.sha(OUT/f'seed_{seed}_predictions.json')==frozen[str(seed)]['sha256']
        correct=sum(pred[k]==labels[k] for k in labels);distance=sum(Levenshtein.distance(labels[k],pred[k]) for k in labels)
        result=dict(seed=seed,correct=correct,total=len(labels),exact_match_percent=100*correct/len(labels),character_error_rate_percent=100*distance/sum(map(len,labels.values())),
            predicted_lengths=dict(collections.Counter(map(len,pred.values()))))
        results.append(result)
    for r in records:details.append(dict(**r,label=labels[r['opaque_id']],predictions={str(s):p[r['opaque_id']] for s,p in predictions.items()}))
    scores=[r['exact_match_percent'] for r in results]
    summary=dict(results=results,mean_exact_match_percent=float(np.mean(scores)),sample_std_percentage_points=float(np.std(scores,ddof=1)),
        evaluation_images=len(labels),excluded_controller_images=100,synthetic_pixel_overlap=0,training_initialization='from_scratch',
        limitation='Historical dataset evaluation; no hyperparameters or checkpoints were selected using these scores.')
    U.dump(OUT/'summary.json',summary);U.dump(OUT/'per_image_results.json',details)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);sheet=Image.new('RGB',(1150,800),'#f5f7fa');d=ImageDraw.Draw(sheet)
    for x,title in [(10,'Real DarkBay'),(260,'Answer'),(450,'Seed48'),(685,'Seed123'),(920,'Seed456')]:d.text((x,12),title,font=font,fill='black')
    for i,row in enumerate(details[:12]):
        y=55+i*60;im=ImageOps.contain(Image.open(row['image_path']),(235,50));sheet.paste(im,(10,y));d.text((260,y+10),row['label'],font=font,fill='black')
        for x,seed in [(450,48),(685,123),(920,456)]:
            text=row['predictions'][str(seed)];d.text((x,y+10),text or '(empty)',font=font,fill='#167540' if text==row['label'] else '#b62525')
    sheet.save(OUT/'examples.png')
    page='<!doctype html><meta charset="utf-8"><title>PARSeq DarkBay test</title><style>body{font:18px system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}img{width:100%}</style><h1>PARSeq DarkBay test</h1><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><p>Examples: first12 images in frozen pixel-hash order, not selected by outcome. Green is correct; red is incorrect.</p><img src="examples.png">'
    (OUT/'report.html').write_text(page,encoding='utf8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
