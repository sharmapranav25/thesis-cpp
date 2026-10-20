"""Six fixed examples per scheme; unchanged custom/DINO/mixed objectives."""
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from . import runtime as U
from .decoder.common import StageLock,now

FAMILIES=['DarkBay','City','CryptBB','SimpleCaptcha','Outlined']
OUT=U.resolve(U.CONFIG['decoder_run'])/'diagnostics/smaller_hybrid_controllers_20261010'
ARMS=['custom','dino','mixed']


def source_predictions(family):
    return U.WORK/('capri_length_cryptbb_100/predictions.json' if family=='CryptBB' else 'capri_length_100/'+family+'/predictions.json')


def checked_refs(family):
    rows=U.read(source_predictions(family));assert len(rows)==100
    refs=[dict(source=r['source'],pixel_sha256=r['pixel_sha256']) for r in rows]
    assert len({r['pixel_sha256'] for r in refs})==100
    for r in refs:
        pixels=np.asarray(Image.open(r['source']).convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256']
    return refs


def reuse_darkbay(refs):
    old=U.WORK/'capri_darkbay_workflow';out=OUT/'DarkBay';out.mkdir()
    assert [r['pixel_sha256'] for r in refs]==[r['pixel_sha256'] for r in U.read(old/'inputs.json')]
    active=U.registry()['models'];old_models=U.read(old/'model_manifest.json')
    assert all(old_models[k]['sha256']==active[k]['sha256'] for k in ['hybrid','custom_encoder','text_mask','dino'])
    U.dump(out/'inputs.json',refs)
    for name in ['length_decision.json','raw_hybrid_predictions.json','refined_recipes.json','pilot.png']:
        shutil.copy2(old/name,out/name)
    U.dump(out/'entries.json',U.read(old/'entries.json')[:18])
    for arm in ARMS:
        (out/arm).mkdir();labels=U.read(old/arm/'labels.json')[:6]
        for row in labels:
            for key in ['image','mask']:
                shutil.copy2(old/row[key],out/row[key])
                assert U.sha(old/row[key])==U.sha(out/row[key])
        U.dump(out/arm/'labels.json',labels)
    U.dump(out/'reuse.json',dict(source=str(old),exact_ordered100_inputs=True,all_four_checkpoint_hashes_match=True,
        original_protocol_sha256=U.sha(old/'protocol.json'),original_model_manifest_sha256=U.sha(old/'model_manifest.json'),
        note='First6 outputs of the completed100-image three-controller run; copied unchanged, not refitted.'))


def audit(family):
    root=OUT/family;refs=U.read(root/'inputs.json');entries=U.read(root/'entries.json');assert len(entries)==18
    prior=U.read(source_predictions(family));raw=U.read(root/'raw_hybrid_predictions.json')
    assert [r['prediction']['length'] for r in raw]==[r['predicted_length'] for r in prior]
    target_hashes={r['pixel_sha256'] for r in refs};hashes=[];strict={}
    for arm in ARMS:
        labels=U.read(root/arm/'labels.json');assert len(labels)==6
        strict[arm]=sum(bool(r['strict']) for r in labels)
        selected=[e for e in entries if e['controller']==arm];assert len(selected)==6
        for row,entry in zip(labels,selected):
            assert row['complete'] and row['usable'] and len(row['text'])==entry['controls']['length']
            pixels=np.asarray(Image.open(root/row['image']).convert('RGB'));digest=hashlib.sha256(pixels.tobytes()).hexdigest()
            assert digest not in target_hashes;hashes.append(digest)
    result=dict(verified_at=now(),target_images=100,refined_inputs=6,examples=18,per_arm=6,complete=18,usable=18,
        strict_per_arm=strict,unique_output_pixels=len(set(hashes)),raw_count_predictions_match_saved=True,
        real_pixels_copied=False,length_decision=U.read(root/'length_decision.json'),
        reused=family=='DarkBay',limitation='Fit-set visual examples; no real strings, no reconstruction/reader accuracy claim.')
    U.dump(root/'preview_audit.json',result);return result


def report(results):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    sheet=Image.new('RGB',(1200,70+len(results)*220),'#f4f6f8');d=ImageDraw.Draw(sheet)
    for col,title in enumerate(['Real input','Custom controller','DINO controller','Mixed controller']):d.text((10+300*col,12),title,fill='#182033',font=font)
    row_index=0
    page='<html><meta charset="utf-8"><title>Smaller hybrid through three controllers</title><style>body{font:17px system-ui;max-width:1200px;margin:24px auto}img{width:100%}</style><h1>Smaller hybrid: controller previews</h1><p>First6 original inputs from each existing100-image set. Each controller compares against the full100-image reference distribution. No target strings, VLMs, training or corpus generation. All three outputs are preserved. Synthetic strings are independent of the target. DarkBay reuses its exact-input, checkpoint-verified historical controller outputs.</p><p>Length is the model-voted majority when at least60/100 agree. Otherwise these previews follow the previously specified equal-observed-length fallback, using seeded shuffled cycles. This does not correct a wrong model count using real labels.</p>'
    for name,value in results.items():
        if 'error' in value:page+='<p>'+html.escape(name+': '+value['error'])+'</p>';continue
        root=OUT/name;refs=U.read(root/'inputs.json');labels={a:U.read(root/a/'labels.json') for a in ARMS}
        for i in range(2):
            y=55+row_index*110;d.text((10,y),f'{name} — input {i+1}',fill='#182033',font=font)
            paths=[Path(refs[i]['source']),*[root/labels[a][i]['image'] for a in ARMS]]
            for col,path in enumerate(paths):
                im=ImageOps.contain(Image.open(path).convert('RGB'),(286,76),Image.Resampling.NEAREST)
                sheet.paste(im,(300*col+(300-im.width)//2,y+25))
            row_index+=1
        mode=value['length_decision'].get('mode','majority')
        page+='<h2>'+name+'</h2><p>'+html.escape('Length policy: '+mode+'. Strict readability checks passed: '+str(value['strict_per_arm'])+' (out of6 per controller). These checks are not target similarity accuracy.')+'</p><img src="'+name+'/pilot.png">'
    sheet.save(OUT/'overview.png');page+='</html>';(OUT/'report.html').write_text(page,encoding='utf8')


def main():
    import psutil
    current=psutil.Process();own={current.pid,*[p.pid for p in current.parents()]}
    for proc in psutil.process_iter(['cmdline']):
        if proc.pid not in own and any(x in (proc.info['cmdline'] or []) for x in ['controllers','preview-smaller-hybrid','timed-pilot-worker','train-baseline']):
            raise RuntimeError('Another controller/training job exists')
    if OUT.exists():raise FileExistsError('Preserving existing preview run')
    for k,entry in U.registry()['models'].items():
        if entry['status']=='active':assert U.sha(U.model_path(k))==entry['sha256']
    OUT.mkdir(parents=True)
    with StageLock(OUT,'preview'):
        U.dump(OUT/'protocol.json',dict(created_at=now(),families=FAMILIES,refined_inputs_per_family=6,reference_inputs_per_family=100,
            selected='First6 saved-manifest entries; no selection by outcome',controllers=ARMS,
            authorization='User requested passing smaller hybrid inputs through controllers, selected first6 per scheme.',
            source_hashes={str(p.relative_to(U.WORK)):U.sha(p) for p in [Path(__file__),U.WORK/'src/capri_controller/pipeline.py',U.WORK/'src/capri_controller/controllers.py']},
            checkpoint_hashes={k:v['sha256'] for k,v in U.registry()['models'].items()},training=False,vlm=False,corpus_generation=False))
        results={}
        for family in FAMILIES:
            refs=checked_refs(family);manifest=OUT/(family+'_targets.json');U.dump(manifest,dict(fit=refs))
            U.dump(OUT/'status.json',dict(state='running',family=family,completed=list(results),updated_at=now()))
            if family=='DarkBay':reuse_darkbay(refs)
            else:
                cmd=[U.CONFIG['python'],'-u',str(U.WORK/'capri.py'),'controllers','--preview','--out',str(OUT/family),'--targets',str(manifest)]
                with (OUT/(family+'.log')).open('w',encoding='utf8') as log:
                    finished=subprocess.run(cmd,cwd=str(U.WORK),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if finished.returncode:
                    results[family]=dict(error='Controller process exited '+str(finished.returncode)+'. See '+family+'.log');report(results);continue
            results[family]=audit(family);U.dump(OUT/'results.json',results);report(results)
            print(family+' complete:18 controller examples',flush=True)
        U.dump(OUT/'results.json',results);report(results)
        U.dump(OUT/'status.json',dict(state='complete' if all('error' not in v for v in results.values()) else 'completed_with_errors',updated_at=now(),completed=list(results),training_started=False))
        print(json.dumps({k:dict(error=v.get('error'),strict=v.get('strict_per_arm')) for k,v in results.items()}),flush=True)
