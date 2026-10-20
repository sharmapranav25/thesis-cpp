from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');dst=Path('joint_mapping_trial');shutil.copy2(root/'feature_multilabel_v1/corpus.py',dst/'corpus.py')
s=(root/'feature_multilabel_v1/train.py').read_text();a=s.index('class Model(');b=s.index('\ndef ap(',a);s=s[:a]+'from models import Joint\nfrom controls import targets,OWNERS,NAMES,LOW,HIGH\n'+s[b:]
s=s.replace("HERE=Path(__file__).parent;OUT=HERE/'run';GRIDS=[(1,1),(2,2),(5,3),(10,3)]","import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['gridded','conformer'],required=True);KIND=parser.parse_args().kind\nHERE=Path(__file__).parent;OUT=HERE/('run_'+KIND);GRIDS=[(10,3)]")
s=s.replace("monitor=ROOT/'monitor/multilabel_status.json'","monitor=ROOT/'monitor'/('joint_'+KIND+'_status.json')")
for old,new in [('204000000','404000000'),('205000000','405000000'),('206000000','406000000'),('207000000','407000000')]:s=s.replace(old,new)
s=s.replace("['corpus.py','train.py','PLAN.md']","['corpus.py','train.py','models.py','controls.py','PLAN.md']")
s=s.replace("protocol=dict(hashes=hashes,","protocol=dict(architecture=KIND,parameter_names=NAMES,parameter_selection='macroAP minus.25normalizedMAE',hashes=hashes,")
s=s.replace("Model().cuda()","Joint(KIND).cuda()")
s=s.replace("model(x[i:min(i+32,n)].float()/255).sigmoid()","model(x[i:min(i+32,n)].float()/255)[0].sigmoid()")
pos=s.index(" x,y,v=data('train');")
s=s[:pos]+''' paramcache={}
 def pdata(part):
  if part not in paramcache:
   records=json.loads((OUT/(part+'_records.json')).read_text());y=data(part)[1];v=data(part)[2]
   paramcache[part]=(np.stack([targets(r['seed'],part) for r in records]),y[:,OWNERS]*v[:,OWNERS])
  return paramcache[part]
 def predict_params(model,part):
  x,_,_=data(part);model.eval();ps=[]
  with torch.inference_mode():
   for i in range(0,len(x),32):ps.append(model(x[i:i+32].float()/255)[1].cpu().numpy())
  return np.concatenate(ps)
 '''.rstrip()+ '\n'+s[pos:]
s=s.replace("confirm=[]","confirm=[];pt,pv=pdata('train');ptt=torch.tensor(pt,device='cuda');pvt=torch.tensor(pv,device='cuda');medians=np.array([np.median(pt[pv[:,i]>0,i]) for i in range(len(NAMES))]);write('parameter_baseline.json',dict(normalized_median=medians.tolist(),names=NAMES))")
s=s.replace("logits=model(a);loss=","logits,parameters=model(a);classification_loss=")
s=s.replace("/vt[ids].sum().clamp_min(1);opt.zero_grad()","/vt[ids].sum().clamp_min(1);regression_loss=(F.smooth_l1_loss(parameters,ptt[ids],reduction='none')*pvt[ids]).sum()/pvt[ids].sum().clamp_min(1);loss=classification_loss+.5*regression_loss;opt.zero_grad()")
s=s.replace("score=float(np.mean(scores));history.append", "st,sv=pdata('selection');sp=predict_params(model,'selection');mae=float((np.abs(sp-st)*sv).sum()/sv.sum());score=float(np.mean(scores))-.25*mae;history.append")
s=s.replace("macro_ap=score,per_feature_ap=", "selection_score=score,macro_ap=float(np.mean(scores)),parameter_mae=mae,per_feature_ap=")
s=s.replace("selection_macro_ap=score)","selection_macro_ap=float(np.mean(scores)),parameter_mae=mae)")
s=s.replace("selection_macro_ap=best","selection_score=best")
s=s.replace("per[part]=metrics(*predict(model,part),thresholds);del cache[part]", "per[part]=metrics(*predict(model,part),thresholds);target,valid=pdata(part);pred=predict_params(model,part);error=np.abs(pred-target);base=np.abs(medians-target);per[part+'_parameters']={name:dict(normalized_mae=float(error[valid[:,j]>0,j].mean()),median_baseline_mae=float(base[valid[:,j]>0,j].mean()),physical_mae=float(error[valid[:,j]>0,j].mean()*(HIGH[j]-LOW[j]))) for j,name in enumerate(NAMES)};del cache[part]")
s=s.replace("report='# Multi-label v1 results", "report='# Joint '+KIND+' mapping results")
s=s.replace("(ROOT/'docs/V2_MULTILABEL_RESULTS.md')","(ROOT/'docs'/('V2_JOINT_'+KIND.upper()+'_RESULTS.md'))")
s=s.replace("Multi-label v1 finite run complete. See V2_MULTILABEL_RESULTS.md", "Joint mapping finite run complete: '+KIND+'. See per-run results")
s=s.replace("event('complete',per_feature_gate=passed,production_promoted=False)","event('training_complete',per_feature_gate=passed,production_promoted=False)\n import subprocess,sys\n subprocess.run([sys.executable,str(HERE/'generate.py'),'--kind',KIND],check=True)\n event('complete',per_feature_gate=passed,production_promoted=False)")
(dst/'train.py').write_text(s)
print('Built joint training harness')
