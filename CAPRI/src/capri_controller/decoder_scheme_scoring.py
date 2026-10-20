"""Post-freeze count scoring; never changes predictions, rendering or training."""
from pathlib import Path
from .decoder.common import ROOT,read,write,sha,now


def main():
    run=ROOT/'runs/capri_decoder_500k_20261007'
    root=run/'diagnostics/real_three_schemes_100_epoch8_20261008_v2'
    output=root/'count_scoring.json'
    if output.exists():raise FileExistsError('Preserving scoring record')
    freeze=read(root/'prediction_freeze.json');assert sha(root/'predictions.json')==freeze['sha256']
    predictions=read(root/'predictions.json');manifest=read(root/'image_manifests.json')
    config=read(ROOT/'configs/local.json')['decoder_scheme_inputs']
    metadata=read(root/'simple_dataset_metadata.json')
    assert 'always exactly 5 characters' in metadata['description']
    simple_correct=sum(r['count']==5 for r in predictions['SimpleCaptcha'])
    truth={};sources={}
    for split in ['benchmark','test','dev']:
        path=Path(config['city_scoring_splits'])/split/'city.jsonl'
        sources[str(path)]=sha(path)
        for line in path.read_text().splitlines():
            import json
            row=json.loads(line);digest=row['sha256'];count=len(row['label'])
            if digest in truth:assert truth[digest]==count
            truth[digest]=count
    matched=[];unmatched=[]
    for image,pred in zip(manifest['City'],predictions['City']):
        digest=sha(image['source'])
        if digest in truth:matched.append(dict(opaque_id=pred['opaque_id'],image_file_sha256=digest,truth=truth[digest],prediction=pred['count']))
        else:unmatched.append(pred['opaque_id'])
    city_correct=sum(r['truth']==r['prediction'] for r in matched)
    city_conditional=sum(r['count']==5 for r in predictions['City'])
    crypt=read(root/'summary.json')['CryptBB_scoring']['count']
    dark=read(run/'diagnostics/real_darkbay_epoch8_20261008/summary.json')['count']
    result=dict(scored_at=now(),prediction_sha256=freeze['sha256'],code_sha256=sha(__file__),
        CryptBB=dict(correct=crypt['correct'],n=crypt['n'],accuracy=crypt['correct']/crypt['n'],basis='Matched per-image CSV labels after freeze'),
        SimpleCaptcha=dict(correct=simple_correct,n=100,accuracy=simple_correct/100,basis='Publisher specifies exactly five characters per image; used only for scoring',
            source='https://www.kaggle.com/datasets/bjosttveit/simple-captcha',metadata_sha256=sha(root/'simple_dataset_metadata.json')),
        City=dict(correct=city_correct,n=len(matched),accuracy=city_correct/len(matched),unmatched=len(unmatched),
            basis='Exact original file SHA256 match to existing per-image annotations',
            conditional_full100=dict(correct=city_conditional,n=100,accuracy=city_conditional/100,
                assumption='All100 images have five characters, consistent with999 existing City annotations;49 sample annotations not matched'),
            matched_records=matched,unmatched_opaque_ids=unmatched,annotation_sources=sources),
        DarkBay=dict(correct=dark['correct'],n=dark['n'],accuracy=dark['correct']/dark['n'],basis='Previous900-image diagnostic, not a new100-image test'),
        limitations='Count accuracy only. Fixed-length datasets do not establish variable-length generalization; no text/font/box accuracy claim.',
        predictions_changed=False,training_started=False)
    assert sha(root/'predictions.json')==freeze['sha256'];write(output,result)
    print({k:{a:b for a,b in v.items() if a in ['correct','n','accuracy','unmatched','conditional_full100']} for k,v in result.items() if k in ['CryptBB','City','SimpleCaptcha','DarkBay']})
