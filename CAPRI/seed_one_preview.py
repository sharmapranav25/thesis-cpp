import sys, json, shutil, hashlib, subprocess, time, os
from pathlib import Path
ROOT=Path(r'E:\pranav\Final sem\sourcefree_synth')
HERE=ROOT/'feature_joint_mapping_01'
sys.path.insert(0,str(HERE))
sys.argv=[sys.argv[0],'--kind','gridded']
import train as training
import numpy as np
import torch
from models import Joint
from controls import targets,NAMES,OWNERS,HIGH,LOW
from corpus import FEATURES
torch.set_num_threads(2)
OUT=HERE/'seed731_preview'
OUT.mkdir(exist_ok=False)
def status(stage,**kw):
    row=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw)
    (OUT/'status.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
try:
 for kind in ['gridded','conformer']:
    src=HERE/('run_'+kind); dst=OUT/('run_'+kind);dst.mkdir()
    for name in ['model_731.pt','parameter_baseline.json']:
        shutil.copy2(src/name,dst/name)
    digest=hashlib.sha256((dst/'model_731.pt').read_bytes()).hexdigest()
    assert digest==hashlib.sha256((src/'model_731.pt').read_bytes()).hexdigest()
    (dst/'freeze.json').write_text(json.dumps(dict(seed=731,sha256=digest,preliminary=True,training_changes_allowed=False),indent=2))
    net=Joint(kind).cuda();net.load_state_dict(torch.load(dst/'model_731.pt',weights_only=True));net.eval()
    def predict(part):
        d=np.load(src/(part+'.npz'));ps=[];qs=[]
        with torch.inference_mode():
            for i in range(0,len(d['images']),16):
                x=torch.tensor(d['images'][i:i+16].transpose(0,3,1,2).copy(),device='cuda').float()/255
                p,q=net(x);ps.append(p.sigmoid().cpu().numpy());qs.append(q.cpu().numpy())
        return d['labels'],np.concatenate(ps),d['valid'],np.concatenate(qs)
    status('calibration',kind=kind)
    y,p,v,_=predict('calibration');ths=[]
    for k in range(len(FEATURES)):
        ids=v[:,k]>0;t=y[ids,k].astype(bool);scores=p[ids,k];choices=[]
        for th in np.unique(np.r_[scores,1.000001]):
            a=scores>=th;tp=(a&t).sum();fp=(a&~t).sum();fn=(~a&t).sum()
            if fp/max(1,(~t).sum())<=.1:choices.append((2*tp/max(1,2*tp+fp+fn),-float(th)))
        ths.append(-max(choices)[1])
    per={'thresholds':dict(zip(FEATURES,ths))};med=np.array(json.loads((dst/'parameter_baseline.json').read_text())['normalized_median'])
    parts=['locked_same','locked_font','locked_combinations','locked_recipe']
    for part in parts:
        status('evaluation',kind=kind,partition=part)
        y,p,v,q=predict(part);per[part]=training.metrics(y,p,v,ths)
        records=json.loads((src/(part+'_records.json')).read_text());truth=np.stack([targets(r['seed'],part) for r in records]);valid=y[:,OWNERS]*v[:,OWNERS]
        per[part+'_parameters']={n:dict(normalized_mae=float(np.abs(q-truth)[valid[:,j]>0,j].mean()),median_baseline_mae=float(np.abs(med-truth)[valid[:,j]>0,j].mean()),physical_mae=float(np.abs(q-truth)[valid[:,j]>0,j].mean()*(HIGH[j]-LOW[j]))) for j,n in enumerate(NAMES)}
    passed={f:all(per[s][f]['f1']>=.85 and per[s][f]['recall']>=.85 and per[s][f]['fpr']<=.1 for s in parts) for f in FEATURES}
    pp={n:all(per[s+'_parameters'][n]['normalized_mae']<per[s+'_parameters'][n]['median_baseline_mae'] for s in parts) for n in NAMES}
    result=dict(results={'731':per},per_feature_gate=passed,parameter_beats_median_all_tests=pp,preliminary_single_seed=True,production_promoted=False)
    (dst/'results.json').write_text(json.dumps(result,indent=2));(dst/'confirmation.json').write_text(json.dumps([dict(seed=731,selection_score=0,selection='user requested seed731 only')]))
    del net;torch.cuda.empty_cache()
    source=(HERE/'generate.py').read_text().replace("here=Path(__file__).parent","here=Path("+repr(str(OUT))+")")
    script=dst/'generate_preview.py';script.write_text("import sys\nsys.path.insert(0,"+repr(str(HERE))+ ")\n"+source)
    status('real_inference_and_generation',kind=kind)
    subprocess.run([sys.executable,str(script),'--kind',kind],check=True)
    assert digest==hashlib.sha256((src/'model_731.pt').read_bytes()).hexdigest()
    report='# Seed731 '+kind+' preliminary evaluation\n\nFrozen independent snapshot; no training or model selection changes. Full real inference uses existing CV/font profiles; real accuracy is unverified.\n\nPassed synthetic feature gates: '+str([f for f,b in passed.items() if b])+'.\n\nParameters beating median in every stratum: '+str([n for n,b in pp.items() if b])+'.\n'
    (dst/'RESULTS.md').write_text(report)
    status('architecture_complete',kind=kind,feature_gate=passed,parameter_gate=pp)
 status('complete')
except BaseException as exc:
 status('failed',error=repr(exc));raise
