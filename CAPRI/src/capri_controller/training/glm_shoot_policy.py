"""Synthetic calibration of GLM-OCR rejection, followed by frozen real testing."""
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile
from .glm_ocr import WORK, MODEL, SOURCE, read, dump, sha

OUT=WORK/'runs/glm_ocr_synthetic_shoot_policy'
DS2=WORK/'runs/Darkbay_ds2_noFont-intel'
SPLIT=WORK/'runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy/frozen_policy.json'
TARGETS=[.90,.95,.98,.99]


def wilson(good,n):
    if not n:return 0.
    z=1.959963984540054;p=good/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def stats(rows):
    n=len(rows);good=sum(r['correct'] for r in rows)
    return dict(attempts=n,correct=good,accuracy_percent=100*good/n if n else None)


def main():
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor,GlmOcrForConditionalGeneration
    from rapidfuzz.distance import Levenshtein
    torch.set_num_threads(2);torch.manual_seed(48);torch.cuda.set_per_process_memory_fraction(.30)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise FileExistsError('Completed run exists')
    split=read(SPLIT);fit_ids=set(split['discovery_indices']);cal_ids=set(split['calibration_indices'])
    assert len(fit_ids)==len(cal_ids)==2500 and not fit_ids&cal_ids
    curriculum=read(DS2/'curriculum_samples.json')
    assert sha(DS2/'curriculum_samples.json')==read(DS2/'bucket_summary.json')['curriculum_sha256']
    valid=[r for r in curriculum if r['split']=='validation']
    assert {r['index'] for r in valid}==fit_ids|cal_ids
    metadata={r['index']:r for r in read(DS2/'samples.json')}
    receipt=read(MODEL/'download_receipt.json')
    for r in receipt['files']:assert sha(MODEL/r['file'])==r['sha256']
    protocol=dict(model=receipt['repository'],revision=receipt['revision'],synthetic_images=5000,
        discovery=2500,calibration=2500,split_sha256=sha(SPLIT),batch_size=8,
        prompt='Text Recognition:',generation='greedy, max_new_tokens64, original RGB, official processor, BF16 SDPA',
        confidence='Product of generated non-special-token probabilities from normalized generation scores, EOS excluded; tokenization differs from PARSeq characters',
        risk='Joint token confidence times product of Laplace-smoothed emitted-character correctness learned on discovery split; heuristic ranking score',
        thresholds='For targets90/95/98/99%, maximum coverage with >=100 calibration attempts and descriptive Wilson95 lower bound meeting target',
        label_boundary='Synthetic labels only for discovery/calibration after synthetic inference; real labels only after policy and real attempt decisions are frozen',
        limitations=['Post-selection Wilson bounds are descriptive, not guarantees','Previously examined historical real set','Batch8 may numerically differ from prior batch1 baseline; report replay differences','Unknown pretrained-data overlap'],code_sha256=sha(Path(__file__)))
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:dump(OUT/'protocol.json',protocol)
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    processor.tokenizer.padding_side='left'
    model=GlmOcrForConditionalGeneration.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);special=set(processor.tokenizer.all_special_ids)
    def infer(records,load_image,phase):
        journal=OUT/(phase+'_predictions.jsonl');rows=[]
        if journal.exists():rows=[json.loads(line) for line in journal.read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in records[:len(rows)]]
        started=time.time()
        with journal.open('a',encoding='utf8',buffering=1) as f,torch.inference_mode():
            for start in range(len(rows),len(records),8):
                chunk=records[start:start+8]
                messages=[[{'role':'user','content':[{'type':'image','image':load_image(r)},{'type':'text','text':'Text Recognition:'}]}] for r in chunk]
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to('cuda')
                outputs=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64,return_dict_in_generate=True,output_scores=True)
                generated=outputs.sequences[:,inputs['input_ids'].shape[-1]:]
                token_logs=torch.stack([s.float().log_softmax(-1).gather(1,generated[:,i:i+1]).squeeze(1) for i,s in enumerate(outputs.scores)],dim=1).cpu().tolist()
                for r,ids,logs in zip(chunk,generated.cpu().tolist(),token_logs):
                    emitted=[];probs=[];ended=False
                    for token,logp in zip(ids,logs):
                        if token in eos:ended=True;break
                        if token not in special:
                            assert math.isfinite(logp)
                            emitted.append(token);probs.append(logp)
                    raw=processor.decode(emitted,skip_special_tokens=True)
                    row=dict(id=r['id'],prediction=raw.strip(),raw_prediction=raw,token_ids=emitted,token_log_probabilities=probs,
                        joint_confidence=math.exp(sum(probs)) if probs else 0.,hit_token_limit=not ended and len(ids)>=64)
                    f.write(json.dumps(row)+'\n');rows.append(row)
                if start%200==0 or len(rows)==len(records):
                    state=dict(stage=phase,completed=len(rows),total=len(records),seconds=time.time()-started,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
                    dump(OUT/'status.json',state);print(state,flush=True)
        dump(OUT/(phase+'_predictions.json'),rows)
        dump(OUT/(phase+'_freeze.json'),dict(sha256=sha(OUT/(phase+'_predictions.json')),count=len(rows)))
        return rows
    archives={}
    def synth_image(r):
        m=metadata[r['id']]
        if m['shard'] not in archives:archives[m['shard']]=zipfile.ZipFile(DS2/'corpus'/m['shard'])
        im=Image.open(io.BytesIO(archives[m['shard']].read(m['image']))).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==m['pixel_sha256']
        return im
    synthetic=infer([dict(id=r['index']) for r in valid],synth_image,'synthetic')
    for z in archives.values():z.close()
    # Synthetic outputs fixed; now learn character reliability using discovery only.
    truths={r['index']:r['text'] for r in valid}
    fit=[r for r in synthetic if r['id'] in fit_ids];cal=[r for r in synthetic if r['id'] in cal_ids]
    charstats=collections.defaultdict(lambda:[0,0]);confusions=collections.Counter();deletions=collections.Counter()
    for r in fit:
        truth=truths[r['id']];pred=r['prediction']
        for tag,a,b,c,d in Levenshtein.opcodes(truth,pred):
            if tag=='equal':
                for char in pred[c:d]:charstats[char][0]+=1;charstats[char][1]+=1
            elif tag=='replace':
                for true,char in zip(truth[a:b],pred[c:d]):charstats[char][1]+=1;confusions[(true,char)]+=1
            elif tag=='insert':
                for char in pred[c:d]:charstats[char][1]+=1;confusions[('<insertion>',char)]+=1
            elif tag=='delete':
                for char in truth[a:b]:deletions[char]+=1
    reliability={c:(good+1)/(n+2) for c,(good,n) in charstats.items()}
    def risk(r):return r['joint_confidence']*math.prod(reliability.get(c,.5) for c in r['prediction']) if r['prediction'] else 0.
    for r in synthetic:r.update(score=risk(r),correct=r['prediction']==truths[r['id']])
    ordered=sorted(cal,key=lambda r:-r['score']);good=0;choices=[]
    for i,r in enumerate(ordered):
        good+=r['correct'];n=i+1
        if i+1<len(ordered) and ordered[i+1]['score']==r['score']:continue
        if n>=100:choices.append(dict(threshold=r['score'],attempts=n,lower=wilson(good,n)))
    policies=[]
    for target in TARGETS:
        eligible=[r for r in choices if r['lower']>=target]
        choice=max(eligible,key=lambda r:r['attempts']) if eligible else None
        policies.append(dict(target=target,available=choice is not None,threshold=choice['threshold'] if choice else None))
    dump(OUT/'character_analysis.json',dict(characters={c:dict(correct=g,total=n,reliability=reliability[c]) for c,(g,n) in charstats.items()},
        confusions=[dict(truth=a,predicted=b,count=n) for (a,b),n in confusions.most_common()],deletions=dict(deletions)))
    dump(OUT/'frozen_policy.json',dict(reliability=reliability,unseen_character_reliability=.5,policies=policies,source_synthetic_sha256=sha(OUT/'synthetic_predictions.json')))
    policy_sha=sha(OUT/'frozen_policy.json');dump(OUT/'policy_freeze.json',dict(sha256=policy_sha))
    synthetic_results=[]
    for p in policies:
        selected=[r for r in cal if p['available'] and r['score']>=p['threshold']]
        synthetic_results.append(dict(**p,**stats(selected),shoot_rate_percent=100*len(selected)/2500))
    dump(OUT/'synthetic_summary.json',dict(discovery=stats(fit),calibration_baseline=stats(cal),policies=synthetic_results))
    records=read(SOURCE/'image_manifest.json');assert len(records)==900
    def real_image(r):
        im=Image.open(r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        return im
    real=infer([dict(**r,id=r['opaque_id']) for r in records],real_image,'real')
    for r in real:
        r['score']=risk(r);r['attempt']=[p['available'] and r['score']>=p['threshold'] for p in policies]
    dump(OUT/'real_decisions.json',real);decisions_sha=sha(OUT/'real_decisions.json')
    dump(OUT/'real_decision_freeze.json',dict(sha256=decisions_sha,policy_sha256=policy_sha))
    # Real labels are accessed only after all attempt decisions have been frozen.
    labels=read(SOURCE/'scoring_labels.json')
    for r in real:r['correct']=r['prediction']==labels[r['id']]
    results=[]
    for i,p in enumerate(policies):
        selected=[r for r in real if r['attempt'][i]]
        results.append(dict(**p,**stats(selected),shoot_rate_percent=100*len(selected)/900))
    original={r['opaque_id']:r['prediction'] for r in read(WORK/'runs/glm_ocr_darkbay_20261005/predictions.json')}
    summary=dict(synthetic_discovery=stats(fit),synthetic_calibration_baseline=stats(cal),synthetic_policies=synthetic_results,
        real_baseline=stats(real),real_policies=results,changed_predictions_vs_original_batch1=sum(r['prediction']!=original[r['id']] for r in real),
        token_limit_counts=dict(synthetic=sum(r['hit_token_limit'] for r in synthetic),real=sum(r['hit_token_limit'] for r in real)),
        policy_sha256=policy_sha,limitation='Synthetic-derived fixed rules; no real-guided threshold adjustment. Batch8 confidence run is separately recorded.')
    assert sha(OUT/'frozen_policy.json')==policy_sha and sha(OUT/'real_decisions.json')==decisions_sha
    dump(OUT/'summary.json',summary);dump(OUT/'status.json',dict(stage='complete'));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
