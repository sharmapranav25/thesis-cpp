"""Meta-Sim-inspired distribution matching, not full paper reproduction."""
import os,sys,json,time,hashlib,copy,math
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).parent;PKG=ROOT/'CAPRI';OUT=ROOT/'capri_metasim_01'
sys.path.insert(0,str(PKG));import capri
capri.verify();torch.set_num_threads(2);cv2.setNumThreads(1)
DINO=Path(r'C:\Users\pranav\.cache\torch\hub\facebookresearch_dino_main');sys.path.insert(0,str(DINO));import vision_transformer as vits
WEIGHTS=Path(r'C:\Users\pranav\.cache\torch\hub\checkpoints\dino_deitsmall8_pretrain.pth')
encoder=vits.vit_small(patch_size=8,num_classes=0);encoder.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));encoder.cuda().eval();encoder.requires_grad_(False)
OUT.mkdir(exist_ok=False)
source=(PKG/'renderer.py').read_text();old="color=choices[int(rng.integers(len(choices)))]";assert source.count(old)==1
source=source.replace(old,"color=choices[int(rng.integers(len(choices)))];color=style.get('locked_glyph_colors',[r for r in []])[i] if 'locked_glyph_colors' in style else color")
(OUT/'renderer.py').write_text(source);scope={};exec(compile(source,str(OUT/'renderer.py'),'exec'),scope);render=scope['render'];fonts=json.loads((PKG/'base/assets/fonts.json').read_text())
NAMES=['line_count_gain','line_width_gain','dot_count_gain','curve_count_gain','curve_amplitude_gain','block_count_gain']
def status(stage,**kw):
 row=dict(stage=stage,pid=os.getpid(),time=time.time(),**kw);(OUT/'status.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def features(images):
 tensors=[np.array(im.convert('RGB').resize((224,80),Image.Resampling.BILINEAR)).transpose(2,0,1) for im in images];result=[]
 with torch.inference_mode():
  for i in range(0,len(tensors),32):
   x=torch.tensor(np.stack(tensors[i:i+32]),device='cuda').float()/255;x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];result.append(torch.nn.functional.normalize(encoder(x),dim=1).cpu().numpy())
 return np.concatenate(result)
def diagnostics(images):
 rows=[]
 for im in images:
  a=np.array(im.convert('RGB').resize((192,96)));g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);edge=(cv2.Canny(g,80,160)>0).astype(float);rows.append(np.r_[a.mean((0,1))/255,a.std((0,1))/255,edge.mean(),np.abs(np.diff(g.astype(float),axis=0)).mean()/255,np.abs(np.diff(g.astype(float),axis=1)).mean()/255,edge.reshape(3,32,192).mean((1,2))])
 return np.array(rows)
def mmd(x,y,bw):
 return torch.exp(-torch.cdist(x,x).square()/(2*bw)).mean()+torch.exp(-torch.cdist(y,y).square()/(2*bw)).mean()-2*torch.exp(-torch.cdist(x,y).square()/(2*bw)).mean()
class Adapter(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(8,24),torch.nn.Tanh(),torch.nn.Linear(24,6));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return math.log(4)*torch.tanh(self.net(x))
def context(entry):
 s=entry['style'];p=entry['joint_prediction']['controls'];return [s['background_edge_fraction'],min(s['small_component_density']*100,1),s.get('texture_fraction',0),s['text_height'],p['curve_count']/4,p['curve_amplitude']/24,min(s.get('background_block_evidence',{}).get('count',0)/35,1),float(s['dark_background'])]
def cases(entries,n,seed,profile):
 rng=np.random.default_rng(seed);result=[]
 for i in range(n):
  e=entries[int(rng.integers(len(entries)))];text,_=capri.pipeline.make_text(rng,['numeric','alphabetic','alphanumeric'][i%3],2+(i//3)%7,'train');s=copy.deepcopy(e['style']);rs=int(rng.integers(2**31-1));im,mask,r=render(s,text,str(PKG/'base'/fonts[profile['font_index']]['path']),rs,profile['variant']);s['locked_glyph_colors']=[g['color'] for g in r['glyphs']]
  result.append(dict(style=s,text=text,seed=rs,font=profile['font_index'],variant=profile['variant'],baseline=im,recipe=r,context=context(e)))
 return result
def adjusted(case,z):
 s=copy.deepcopy(case['style']);g=np.exp(np.asarray(z));s['line_count_gain']=s.get('line_count_gain',1)*g[0];s['line_width_gain']=s.get('line_width_gain',1)*g[1];s['dot_count_gain']=s.get('dot_count_gain',1)*g[2];s['curve_count_gain']=s.get('curve_count_gain',1)*g[3]
 if 'pred_curve_amplitude' in s:s['pred_curve_amplitude']*=g[4]
 else:s['curve_amplitude_gain']=s.get('curve_amplitude_gain',1)*g[4]
 ev=s.get('background_block_evidence',{})
 if ev.get('state')=='supported':ev['count']=max(0,round(ev['count']*g[5]))
 try:
  im,mask,r=render(s,case['text'],str(PKG/'base'/fonts[case['font']]['path']),case['seed'],case['variant'])
  assert [(v['character'],v['box'],v['angle'],v['color']) for v in r['glyphs']]==[(v['character'],v['box'],v['angle'],v['color']) for v in case['recipe']['glyphs']]
  return im,r,False
 except ValueError:
  return case['baseline'],case['recipe'],True
def evaluate(model,cs,target,bw):
 with torch.no_grad():z=model(torch.tensor([c['context'] for c in cs],dtype=torch.float32)).numpy()
 vals=[adjusted(c,a) for c,a in zip(cs,z)];f=features([v[0] for v in vals]);return float(mmd(torch.tensor(f),target,bw)),vals,z
protocol=dict(method='Meta-Sim-inspired residual MLP attribute transformer + Gaussian feature MMD + finite differences through renderer',paper='https://arxiv.org/abs/1904.11621',borrowed_sections=['3.1 mutable attributes','3.2.1 identity initialization','3.2.2 MMD and finite differences'],deviations='MLP on flat CAPRI recipe, not GCN scene graph; DINO ViTS8 not Inception; identity residual init instead of autoencoder pretraining; no supervised task meta-objective; no exact reproduction claim',models_frozen=True,real_labels=False,runtime_vlm=False,controls=NAMES,gain_bounds=[.25,4],steps=12,batch=4,finite_difference_epsilon=.2,selection_every=4,fit=48,selection=16,report=32,report_synthetic=64,historical_real_exposure=True,split='sort unique decoded pixel SHA256 with fixed salt; no report recipe used for generation; cached font selected historically using broader pool',colour_policy='No palette/gradient/font/layout/text changes; exact baseline glyph RGB/geometry assertion; appearance histograms may change with primitive coverage',dino_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),capri_manifest_sha256=hashlib.sha256((PKG/'INTEGRITY.json').read_bytes()).hexdigest(),evaluation='DINO MMD and independent CV standardized mean discrepancy; no OCR utility claim or automatic promotion')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));allresults={}
try:
 for fi,family in enumerate(['cryptbb','city','courier','darkbay','outlined_text_captcha']):
  status('preparing',family=family);folder=OUT/family;folder.mkdir();profile=json.loads((PKG/'profiles/conformer'/(family+'.json')).read_text());unique={}
  for e in profile['entries']:
   im=Image.open(e['reference_path_provenance']).convert('RGB');digest=hashlib.sha256(np.array(im).tobytes()+str(im.size).encode()).hexdigest();unique[digest]=e
  ordered=sorted(unique,key=lambda h:hashlib.sha256(('capri-meta-v1:'+h).encode()).hexdigest());assert len(ordered)>=96
  splits={'fit':ordered[:48],'selection':ordered[48:64],'report':ordered[64:96]};(folder/'splits.json').write_text(json.dumps({k:[dict(pixel_sha256=h,path=unique[h]['reference_path_provenance']) for h in v] for k,v in splits.items()},indent=2))
  fitentries=[unique[h] for h in splits['fit']];real={k:[Image.open(unique[h]['reference_path_provenance']).convert('RGB') for h in v] for k,v in splits.items() if k!='report'}
  fit=torch.tensor(features(real['fit']));selection=torch.tensor(features(real['selection']));bw=max(.01,float(torch.pdist(fit).square().median()));torch.manual_seed(800+fi);model=Adapter();optimizer=torch.optim.Adam(model.parameters(),lr=.035);validcases=cases(fitentries,16,990+fi,profile)
  best,_,_=evaluate(model,validcases,selection,bw);baseline_selection=best;torch.save(model.state_dict(),folder/'adapter.pt');history=[];bad=0
  for step in range(12):
   cs=cases(fitentries,4,10000+fi*100+step,profile);x=torch.tensor([c['context'] for c in cs],dtype=torch.float32);z=model(x);zn=z.detach().numpy();vals=[adjusted(c,a) for c,a in zip(cs,zn)];ff=features([v[0] for v in vals]);ft=torch.tensor(ff,requires_grad=True);loss=mmd(ft,fit,bw);loss.backward();jac=[]
   for j in range(6):
    plus=zn.copy();minus=zn.copy();plus[:,j]+=.2;minus[:,j]-=.2
    pv=[adjusted(c,a) for c,a in zip(cs,plus)];mv=[adjusted(c,a) for c,a in zip(cs,minus)];bad+=sum(v[2] for v in vals+pv+mv);f=features([v[0] for v in pv+mv]);jac.append((f[:4]-f[4:])/.4)
   grad=np.stack([(ft.grad.numpy()*v).sum(1) for v in jac],1);optimizer.zero_grad();z.backward(torch.tensor(grad)+.001*z.detach()/z.numel());torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();row=dict(step=step+1,fit_mmd=float(loss.detach()))
   if (step+1)%4==0:
    score,_,_=evaluate(model,validcases,selection,bw);row['selection_mmd']=score
    if score<best:best=score;torch.save(model.state_dict(),folder/'adapter.pt')
   history.append(row);(folder/'history.json').write_text(json.dumps(history,indent=2));status('training',family=family,**row)
  # Freeze selection before opening report pixels. No further adaptation afterward.
  freeze=hashlib.sha256((folder/'adapter.pt').read_bytes()).hexdigest();(folder/'freeze.json').write_text(json.dumps(dict(sha256=freeze,selection_mmd=best,baseline_selection_mmd=baseline_selection)))
  model.load_state_dict(torch.load(folder/'adapter.pt',weights_only=True));model.eval();report=[Image.open(unique[h]['reference_path_provenance']).convert('RGB') for h in splits['report']];rf=features(report);cs=cases(fitentries,64,40000+fi,profile);base=[c['baseline'] for c in cs];after_score,vals,z=evaluate(model,cs,torch.tensor(rf),bw);after=[v[0] for v in vals];bf=features(base);af=features(after);base_score=float(mmd(torch.tensor(bf),torch.tensor(rf),bw));sd=np.maximum(diagnostics(real['fit']).std(0),.03);rd=diagnostics(report).mean(0);bc=float(np.mean(np.abs(diagnostics(base).mean(0)-rd)/sd));ac=float(np.mean(np.abs(diagnostics(after).mean(0)-rd)/sd))
  # Paired bootstrap uses identical synthetic indices for before/after and shared real indices.
  rng=np.random.default_rng(550+fi);deltas=[]
  for _ in range(300):
   ix=rng.integers(0,64,64);iy=rng.integers(0,32,32);r=torch.tensor(rf[iy]);deltas.append(float(mmd(torch.tensor(af[ix]),r,bw)-mmd(torch.tensor(bf[ix]),r,bw)))
  metrics=dict(baseline_mmd=base_score,refined_mmd=after_score,relative_mmd_reduction=(base_score-after_score)/max(base_score,1e-9),paired_bootstrap_delta_ci95=np.quantile(deltas,[.025,.975]).tolist(),cv_before=bc,cv_after=ac,mean_gains=dict(zip(NAMES,np.exp(z).mean(0).tolist())),final_fallbacks=sum(v[2] for v in vals),finite_difference_invalid_count=bad,checkpoint_sha256=freeze,selection_mmd=best)
  sheet=Image.new('RGB',(900,780),'white');draw=ImageDraw.Draw(sheet)
  for j,label in enumerate(['Real report examples (unpaired)','Frozen CAPRI','Meta-Sim-inspired']):draw.text((j*300+3,4),label,fill='black')
  records=[]
  for i,(c,im,v) in enumerate(zip(cs,after,vals)):
   base[i].save(folder/f'{i:03d}_before.png');im.save(folder/f'{i:03d}_after.png');records.append(dict(index=i,text=c['text'],seed=c['seed'],gain=np.exp(z[i]).tolist(),before=c['recipe'],after=v[1],fallback=v[2]))
   if i<6:
    for j,pic in enumerate([report[i],base[i],im]):
     pic=pic.copy();pic.thumbnail((285,105));sheet.paste(pic,(j*300+3,30+i*125))
  sheet.save(folder/'comparison.png');(folder/'recipes.json').write_text(json.dumps(records,indent=2));(folder/'results.json').write_text(json.dumps(metrics,indent=2));allresults[family]=metrics;(OUT/'results.json').write_text(json.dumps(allresults,indent=2));status('family_complete',family=family,metrics=metrics)
 capri.verify();status('complete',families=5)
except BaseException as exc:status('failed',error=repr(exc));raise
