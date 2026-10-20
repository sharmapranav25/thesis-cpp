"""CAPRI research snapshot: frozen visual inference and procedural recipe fusion."""
import argparse,copy,hashlib,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'base'));sys.path.insert(0,str(ROOT/'model_code'))
import pipeline
from renderer import render
def verify():
 for name,digest in json.loads((ROOT/'INTEGRITY.json').read_text()).items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Integrity mismatch: '+name)
 pipeline.verify()
def fuse(style,pred,kind):
 s=copy.deepcopy(style);s.pop('classifier_features',None);s['multilabel_grid']=False;s['multilabel_blur']=False
 s['classifier_features']={'straight_interference':'present','dot_scatter':'present','curved_interference':'present' if pred['present']['curved_interference'] else 'absent'}
 if pred['present']['curved_interference']:
  p=pred['controls'];scale=s['size'][1]/96;s['curve_count_gain']=p['curve_count']/2;s['pred_curve_width']=p['curve_width']*scale
  if kind=='conformer':s['pred_curve_amplitude']=p['curve_amplitude']*scale
 return s
def main():
 p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
 sp.add_parser('verify')
 g=sp.add_parser('generate');source=g.add_mutually_exclusive_group(required=True);source.add_argument('--family');source.add_argument('--profile');g.add_argument('--kind',choices=['conformer','gridded'],default='conformer');g.add_argument('--out',required=True);g.add_argument('--count',type=int,default=210);g.add_argument('--seed',type=int,required=True);g.add_argument('--partition',choices=['train','validation','test'],required=True);g.add_argument('--mode',choices=['baseline','diverse'],default='baseline');g.add_argument('--masks',action='store_true');g.add_argument('--recipes',action='store_true');g.add_argument('--font-mode',choices=['similar','single'],default='similar');g.add_argument('--font-top-k',type=int,choices=range(1,13),default=3);g.add_argument('--refinement',choices=['auto','on','off'],default='auto')
 q=sp.add_parser('profile');q.add_argument('--references',required=True);q.add_argument('--out',required=True);q.add_argument('--name',required=True);q.add_argument('--kind',choices=['conformer','gridded'],default='conformer');q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--variant',type=int,choices=[0,1,2,3],default=0)
 a=sp.add_parser('audit');a.add_argument('--corpora',nargs='+',required=True);a.add_argument('--evaluation-manifest')
 args=p.parse_args();verify()
 if args.command=='verify':print('CAPRI VERIFIED');return
 if args.command=='audit':pipeline.audit(args);return
 if args.command=='generate':
  if args.family:args.profile=str(ROOT/'profiles'/args.kind/(args.family+'.json'))
  profile=json.loads(Path(args.profile).read_text())
  if profile.get('capri',{}).get('kind')!=args.kind:raise ValueError('Use a CAPRI profile of the selected kind')
  import generation
  from refinement import Refiner
  registry=json.loads((ROOT/'adapters/registry.json').read_text());adapter=registry.get(profile['family']);matched=adapter is not None and adapter['kind']==args.kind and adapter['profile_sha256']==hashlib.sha256(Path(args.profile).read_bytes()).hexdigest()
  if args.refinement=='on' and not matched:raise ValueError('No adapter fitted to this exact profile and architecture; use --refinement off or fit a new adapter separately')
  enabled=args.refinement!='off' and matched
  generation.render=Refiner(ROOT/adapter['checkpoint'],render) if enabled else render
  print('Font mode:',args.font_mode,'Refinement:', 'on' if enabled else 'off (disabled or no matching fitted adapter)')
  generation.generate(args)
  protocol=Path(args.out)/'protocol.json';data=json.loads(protocol.read_text());data['capri']=profile['capri'];data['font_mode']=args.font_mode;data['font_top_k']=args.font_top_k;data['refinement_enabled']=enabled;data['refinement_request']=args.refinement;data['version']='CAPRI-v2-font-mixture-metasim';data['capri_package_manifest_sha256']=hashlib.sha256((ROOT/'INTEGRITY.json').read_bytes()).hexdigest();protocol.write_text(json.dumps(data,indent=2));return
 # Fresh CV and frozen legacy/font inference first; joint inference then fuses recipes.
 subprocess.run([sys.executable,str(ROOT/'base/pipeline.py'),'profile','--references',args.references,'--out',args.out,'--name',args.name,'--device',args.device,'--variant',str(args.variant)],check=True)
 import torch,numpy as np
 from PIL import Image
 from models import Joint
 from controls import decode
 torch.set_num_threads(4);net=Joint(args.kind).to(args.device);net.load_state_dict(torch.load(ROOT/'checkpoints'/args.kind/'model_731.pt',map_location=args.device,weights_only=True));net.eval()
 results=json.loads((ROOT/'checkpoints'/args.kind/'results.json').read_text());thresholds=results['results']['731']['thresholds'];features=list(thresholds)
 path=Path(args.out)/'profile.json';profile=json.loads(path.read_text());path.rename(Path(args.out)/'cv_profile.json')
 with torch.inference_mode():
  for e in profile['entries']:
   im=Image.open(e['reference_path_provenance']).convert('RGB').resize((192,96),Image.Resampling.BILINEAR);x=torch.tensor(np.array(im).transpose(2,0,1).copy(),device=args.device).float()[None]/255;l,c=net(x);scores=dict(zip(features,map(float,l.sigmoid()[0].cpu())));pred=dict(scores=scores,present={f:scores[f]>=thresholds[f] for f in features},controls=decode(c[0].cpu().numpy()));e['joint_prediction']=pred;e['style']=fuse(e['style'],pred,args.kind)
 profile['capri']=dict(kind=args.kind,seed=731,mode='CV-preserving fusion',real_fitting=False,experimental=True);path.write_text(json.dumps(profile,indent=2));print('CAPRI PROFILE',path)
if __name__=='__main__':main()
