"""Post-training real DarkBay evaluation, prediction ensemble and shoot rate."""
import collections
import copy
import gc
import hashlib
import html
import json
import math
import io
from pathlib import Path
from .ds3 import ROOT, DATA, DS1_RUN, SEEDS, U, bootstrap
from .ds3_selection import freeze_selection


def main():
    bootstrap()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    from .ds2_data import transform
    torch.set_num_threads(8)
    out=ROOT/'evaluation_darkbay'
    if out.exists():raise FileExistsError('Preserve evaluation: '+str(out))
    out.mkdir()
    manifest=U.read(DS1_RUN/'evaluation_darkbay/image_manifest.json')
    assert len(manifest)==900
    completed=freeze_selection()
    U.dump(out/'protocol.json',dict(checkpoints=completed['seeds'],images=900,
        selection=completed['selection'],
        decoding='FP32, normal saved autoregressive decoder, one refinement, no length forcing or case changes',
        metric='Case-sensitive exact whole-string accuracy and CER',
        manifest_sha256=U.sha(DS1_RUN/'evaluation_darkbay/image_manifest.json'),
        limitation='Historical DarkBay set already examined for DS1; this is not untouched evaluation'))
    tf=transform(); tensors=[]
    for r in manifest:
        im=Image.open(r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        tensors.append(tf(im))
    x_all=torch.stack(tensors);predictions={};frozen={}
    for run in completed['seeds']:
        seed=run['seed'];assert U.sha(run['checkpoint'])==run['sha256']
        checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(checkpoint['config'])
        decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.cuda().eval()
        rows=[]
        with torch.inference_mode():
            for start in range(0,900,64):
                p=model(x_all[start:start+64].cuda());assert torch.isfinite(p).all()
                decoded=decoder(p);probs=p.cpu().numpy()
                for r,pp,(text,conf) in zip(manifest[start:start+64],probs,decoded):
                    cp=[]
                    for index,prob in zip(pp.argmax(-1),pp.max(-1)):
                        char=decoder.character[int(index)]
                        if char==decoder.EOS:break
                        if char not in (decoder.BOS,decoder.PAD):cp.append(float(prob))
                    assert len(cp)==len(text)
                    rows.append(dict(opaque_id=r['opaque_id'],prediction=text,confidence=float(conf) if math.isfinite(conf) else None,joint_confidence=float(np.prod(cp)) if cp else 0.0))
        name=f'seed_{seed}';predictions[name]=rows
        U.dump(out/(name+'_predictions.json'),rows);frozen[name]=U.sha(out/(name+'_predictions.json'))
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    ensemble=[]
    for i,r in enumerate(manifest):
        candidates=[predictions[f'seed_{s}'][i] for s in SEEDS]
        text,count=collections.Counter(c['prediction'] for c in candidates).most_common(1)[0]
        if count==1:text=max(candidates,key=lambda c:c['confidence'] if c['confidence'] is not None else -1)['prediction']
        joint=sum(c['joint_confidence'] for c in candidates if c['prediction']==text)/3
        ensemble.append(dict(opaque_id=r['opaque_id'],prediction=text,joint_confidence=joint))
    predictions['ensemble']=ensemble
    U.dump(out/'ensemble_predictions.json',ensemble);frozen['ensemble']=U.sha(out/'ensemble_predictions.json')
    U.dump(out/'prediction_freeze.json',frozen)
    labels=U.read(DS1_RUN/'evaluation_darkbay/scoring_labels.json')
    assert set(labels)=={r['opaque_id'] for r in manifest}
    results={}
    for name,rows in predictions.items():
        assert U.sha(out/(name+'_predictions.json'))==frozen[name]
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
        distance=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in rows)
        results[name]=dict(correct=correct,total=900,accuracy=100*correct/900,CER=100*distance/sum(map(len,labels.values())))
    summary=dict(model='Model3',results=results,ds1_ensemble_correct=496,ds2_ensemble_correct=565,
        difference_vs_ds1=results['ensemble']['correct']-496,difference_vs_ds2=results['ensemble']['correct']-565,
        limitation='Historical900-image set; selection recorded in evaluation_selection.json. User final-epoch designation followed examination of seed48 real results; training curriculum unchanged.')
    U.dump(out/'summary.json',summary)
    infer_policy()
    from .ds3_policy import main as policy
    policy()
    report()

def infer_policy():
    bootstrap()
    import torch,lmdb
    import numpy as np
    from PIL import Image
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from capri_controller.training.ds2_data import transform
    torch.set_num_threads(4)
    out=ROOT/'synthetic_policy_inference';out.mkdir(exist_ok=True)
    valid=[r for r in U.read(DATA/'curriculum_samples.json') if r['split']=='validation']
    assert len(valid)==5000
    metadata={r['index']:r for r in U.read(DATA/'samples.json')}
    env=lmdb.open(str(DATA/'all_lmdb'),readonly=True,lock=False,readahead=False)
    tf=transform();images=[]
    with env.begin() as txn:
        for r in valid:
            im=Image.open(io.BytesIO(txn.get(f"image-{r['index']:09d}".encode()))).convert('RGB')
            assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==metadata[r['index']]['pixel_sha256']
            images.append(tf(im))
    env.close();images=torch.stack(images)
    for run in freeze_selection()['seeds']:
        seed=run['seed'];p=out/f'seed_{seed}.json';receipt=p.with_suffix('.receipt.json')
        assert U.sha(run['checkpoint'])==run['sha256']
        if receipt.exists():
            assert U.sha(p)==U.read(receipt)['predictions_sha256']
            assert U.read(receipt)['checkpoint_sha256']==run['sha256']
            continue
        ck=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(ck['config']);decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cuda().eval()
        rows=[]
        with torch.inference_mode():
            for start in range(0,5000,64):
                outputs=model(images[start:start+64].cuda());assert torch.isfinite(outputs).all()
                for r,(text,conf) in zip(valid[start:start+64],decoder(outputs)):
                    rows.append(dict(index=r['index'],prediction=text,confidence=float(conf) if math.isfinite(conf) else None))
        U.dump(p,rows);U.dump(receipt,dict(predictions_sha256=U.sha(p),checkpoint_sha256=run['sha256'],epoch=run['epoch'],images=5000,decoding='FP32 batch64, original RGB bicubic32x128; same as real evaluation',code_sha256=U.sha(Path(__file__))))
        print(f'Seed {seed}: 5000 predictions frozen',flush=True)
        del model,ck;gc.collect();torch.cuda.empty_cache()


def report():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=U.read(ROOT/'evaluation_darkbay/summary.json')['results']
    baseline=U.read(DATA/'evaluation_darkbay/summary.json')['results']
    ds1=U.read(DS1_RUN/'evaluation_darkbay/summary.json')
    policy=U.read(ROOT/'synthetic_validation_shoot_policy/summary.json')
    records=freeze_selection()['seeds']
    lines=['# Model 3 adaptive curriculum results', '',
        'Curriculum is the only changed training factor relative to Model 2. Both use exactly the same95,000 optimization and5,000 validation IDs, augmentation, scratch initialization, optimizer, architecture and20epochs. Validation never updates difficulty or sampling.', '',
        '| Reader | Selected epoch | Model1 correct/900 | Model2 correct/900 | Model3 correct/900 | Model3 accuracy | Model3 CER | Change vs Model2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    old_counts={'seed_48':492,'seed_123':439,'seed_456':474,'ensemble':496}
    for name,r in results.items():
        epoch=next((x['epoch'] for x in records if name==f"seed_{x['seed']}"),'per-seed')
        lines.append(f"| {name} | {epoch} | {old_counts[name]} | {baseline[name]['correct']} | {r['correct']} | {r['accuracy']:.4f}% | {r['CER']:.4f}% | {r['correct']-baseline[name]['correct']:+d} |")
    lines += ['', 'Model1 uses fixed epoch20 and differs in corpus, augmentation and split. Model2 uses synthetic validation selection. Model3 evaluation selection: '+records[0]['selection']+'. The original synthetic-selected checkpoints remain preserved. Real DarkBay is a repeatedly examined historical test set; this selection amendment is not an untouched-test decision.', '',
              '## Synthetic-calibrated shoot rate', '',
              '| Model | Actual threshold | Correct / attempted | Accuracy | Shoot rate |',
              '|---|---:|---:|---:|---:|']
    for title,root in [('1',DS1_RUN),('2',DATA),('3',ROOT)]:
        summary=U.read(root/'synthetic_validation_shoot_policy/summary.json')
        for r in summary['real_policies']:
            if r['available']:
                lines.append(f"| {title} | {r['threshold']:.17g} | {r['correct']}/{r['attempts']} | {r['accuracy_percent']:.4f}% | {r['shoot_rate_percent']:.4f}% |")
            else:
                lines.append(f"| {title} | unavailable | — | — | — |")
    lines += ['', 'Policies use the unchanged2,500/2,500 discovery/calibration partitions. Character risks and thresholds were learned on synthetic data, then frozen; real attempt decisions were frozen before real labels were scored. The same validation set also selected checkpoints. Wilson bounds after threshold search are descriptive, and synthetic goals are not promised real accuracy. Historical confidence sweeps are separate and no new real-accuracy threshold search was performed.', '',
              '## Compute and validation', '', '| Seed | Training seconds | Validation seconds | Full-pool rescoring seconds | Selected validation accuracy |', '|---|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for r in records:
        seed=r['seed'];val=U.read(ROOT/f'seed_{seed}/validation_history.json');loss=U.read(ROOT/f'seed_{seed}/loss_history.json')
        chosen=next(x for x in val if x['epoch']==r['epoch'])
        lines.append(f"| {seed} | {r['training_seconds']:.1f} | {r['validation_seconds']:.1f} | {r['rescoring_seconds']:.1f} | {100*chosen['acc']:.4f}% |")
        axes[0].plot([x['epoch'] for x in val],[100*x['acc'] for x in val],label=str(seed))
        axes[1].plot([x['epoch'] for x in loss],[x['mean_training_loss'] for x in loss],label=str(seed))
        axes[2].plot([x['epoch'] for x in val],[x['mean_sequence_nll'] for x in val],label=str(seed))
    for ax,title in zip(axes,['Synthetic validation accuracy (%)','Mean training loss','Validation sequence NLL']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=160);plt.close(fig)
    total_train=sum(r['training_seconds'] for r in records)
    total_val=sum(r['validation_seconds'] for r in records)
    total_rescore=sum(r['rescoring_seconds'] for r in records)
    ds2_seconds=sum(r['seconds'] for r in U.read(DATA/'completion.json')['seeds'])
    compute=dict(model3_training_seconds=total_train,model3_validation_seconds=total_val,
        model3_rescoring_seconds=total_rescore,rescore_overhead_relative_to_training_and_validation=total_rescore/(total_train+total_val),
        model2_recorded_total_seconds=ds2_seconds,
        model3_last_session_seconds=sum(r['seconds_this_session'] for r in records),
        limitation='Component times span saved epochs; last-session wall time excludes earlier sessions if resumed. Hardware load and checkpoint I/O affect wall-clock comparisons.')
    U.dump(ROOT/'compute_comparison.json',compute)
    lines += ['', f"Full-pool rescoring added {total_rescore/3600:.3f} hours, or {100*total_rescore/(total_train+total_val):.2f}% of measured training plus validation time. Model2 recorded total time: {ds2_seconds/3600:.3f} hours. See compute_comparison.json for timing scope and resume limitations."]
    receipts=[]
    for r in records:
        assert U.sha(r['checkpoint'])==r['sha256'] and U.sha(r['final_checkpoint'])==r['final_sha256']
        for epoch in range(1,21):
            path=ROOT/f"seed_{r['seed']}";receipt=U.read(path/f'epoch_{epoch:02d}.receipt.json')
            assert U.sha(path/f'epoch_{epoch}.pth')==receipt['checkpoint_sha256']
            assert U.sha(path/f'adaptive_epoch_{epoch:02d}.npz')==receipt['adaptive_sha256']
            assert receipt['scored_images']==95000
            receipts.append(receipt)
    U.dump(ROOT/'completion_verification.json',dict(epoch_checkpoints=60,full_rescores=60,
        training_images_per_rescore=95000,selected_and_final_hashes_verified=True,
        real_prediction_freeze=U.read(ROOT/'evaluation_darkbay/prediction_freeze.json'),
        policy_freeze=U.read(ROOT/'synthetic_validation_shoot_policy/policy_freeze.json'),
        real_decision_freeze=U.read(ROOT/'synthetic_validation_shoot_policy/real_decision_freeze.json')))
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    page='<!doctype html><meta charset="utf-8"><title>Model 3 results</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap}img{max-width:100%}</style><img src="curves.png"><pre>'+html.escape('\n'.join(lines))+'</pre>'
    (ROOT/'report.html').write_text(page,encoding='utf8')
    log=U.WORK/'docs/MAINTENANCE_LOG.md'
    marker='## Model 3 completed adaptive run — '+ROOT.name
    if marker not in log.read_text(encoding='utf8'):
        with log.open('a',encoding='utf8') as f:
            f.write('\n'+marker+'\n\nAll three seeds completed20epochs with60 full95000-image rescoring passes. Ensemble selection: '+records[0]['selection']+'. Ensemble: '+str(results['ensemble']['correct'])+'/900; Model2:565/900; Model1:496/900. Full checkpoint/prediction/policy/decision hash verification, validation/loss curves, compute times and policy tables are saved in `'+str(ROOT.relative_to(U.WORK)).replace('\\','/')+'/RESULTS.md`. Only curriculum changed in training relative to Model2; the user later changed evaluation checkpoint designation. Real data remain historical and validation was reused for policy calibration.\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
