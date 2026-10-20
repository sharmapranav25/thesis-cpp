"""Train a synthetic-only font matcher, freeze, infer and render development trials."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time,copy,itertools,argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,new_run,digest,load,pixel_hash
from feature_bank.prepare import DATASETS
from style_pipeline.font_match import FontMatcher,synthetic,encode,infer,views
from style_pipeline.render import render
from style_pipeline.monitor import update
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/style_phase2_fonts_01'

def event(message,**kw):update('Phase 2: font matching',message,run=str(OUT),**kw)

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='style_phase2_fonts_01');parser.add_argument('--steps',type=int,default=700)
    parser.add_argument('--test-seed',type=int,default=20000);parser.add_argument('--warmstart');parser.add_argument('--robust-training',action='store_true');parser.add_argument('--contrast-training',action='store_true');parser.add_argument('--geometry-training',action='store_true');args=parser.parse_args()
    if not args.run.startswith('style_phase2_fonts_') or '/' in args.run or '\\' in args.run:raise ValueError('Invalid run name')
    OUT=ROOT/'runs'/args.run
    out=new_run(OUT);start=time.perf_counter();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(5300)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    fonts=read(ROOT/'runs/feature_bank_pilot_01/font_selection.json')+read(ROOT/'resources/style_phase3_fonts_01/manifest.json')
    for f in fonts:assert digest(f['path'])==f['sha256'] and digest(f['license_path'])==f['license_sha256']
    write(out/'fonts.json',fonts)
    for rel in ('style_pipeline/font_match.py','style_pipeline/render.py','style_pipeline/recipes.py','scripts/run_style_phase3.py'):
        dest=out/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    write(out/'protocol.json',dict(seed=5300,fonts=len(fonts),synthetic_train_per_font=256,synthetic_selection_per_font=64,
        synthetic_test_per_font=64,steps=args.steps,test_seed=args.test_seed,warmstart=args.warmstart,robust_training=args.robust_training,contrast_training=args.contrast_training,geometry_training=args.geometry_training,
        robustness_protocol='when enabled: alternating standard/3x-clutter examples in all synthetic splits; training cycles through 4 views; selection/test full-image view',
        batch=64,selection='lowest synthetic selection loss at 100-step checkpoints',
        classifier_training_real_images=False,real_labels=False,runtime_vlm=False,
        real_partitions='existing Phase 2 fit96/development32 only; no reserved evaluation images',
        font_search='top 3 inferred fonts x filled/bold/hollow x width .9/1.1 x height 1/1.25; plus Phase 2 baseline',
        test_claim='synthetic accuracy plus exploratory real development matching; no real font identity labels',
        papers_register='docs/THESIS_PAPERS.md',pipeline_order='Phase1 directly to fonts; distribution refinement deferred',background_recipe_applied=False))
    arrays={};meta={};seen={}
    for part,count,base in [('train',256,0),('selection',64,10000),('test',64,args.test_seed)]:
        xx=[];yy=[];hh=[];texts=set();identities=[]
        for fi,font in enumerate(fonts):
            event(f'Building synthetic {part}: font {fi+1}/{len(fonts)} ({font["family"]})')
            for i in range(count):
                im,h,text=synthetic(font['path'],base+i,clutter=3. if args.robust_training and i%2 else 1.,geometry=args.geometry_training)
                if args.contrast_training:
                    gain=(1.,.65,.35,.15)[i%4]
                    im=Image.fromarray(np.clip(128+(np.asarray(im,dtype=float)-128)*gain,0,255).astype('uint8'))
                xx.append(views(im)[(i//4)%4 if args.contrast_training else i%4] if args.robust_training and part=='train' else encode(im));yy.append(fi);hh.append(h);texts.add(text)
                identities.append(dict(font=fi,seed=base+i,text=text,pixel_sha256=pixel_hash(im)))
        assert all(not texts&previous for previous in seen.values()),'Synthetic text leakage across partitions'
        seen[part]=texts;arrays[part]=(torch.from_numpy(np.stack(xx)).to('cuda'),torch.tensor(yy,device='cuda'),torch.tensor(hh,device='cuda'))
        meta[part]=identities
    write(out/'synthetic_membership.json',meta)
    model=FontMatcher(len(fonts)).cuda();initial_step=0
    if args.warmstart:
        initial=torch.load(args.warmstart,weights_only=True)
        assert initial['fonts']==[f['sha256'] for f in fonts]
        model.load_state_dict(initial['state_dict']);initial_step=initial['step']
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4 if args.warmstart else 7e-4,weight_decay=1e-4)
    rng=np.random.default_rng(5300);best=float('inf');history=[]
    def evaluate(part):
        x,y,h=arrays[part];pa=[];pb=[];losses=[]
        model.eval()
        with torch.no_grad():
            for i in range(0,len(x),128):
                a,b,_=model(x[i:i+128]);pa.append(a);pb.append(b)
                losses.append(float(F.cross_entropy(a,y[i:i+128])+.4*F.cross_entropy(b,h[i:i+128])))
        a=torch.cat(pa);b=torch.cat(pb);correct=a.argmax(1)==y
        return dict(loss=float(np.mean(losses)),font_top1=float(correct.float().mean()),font_top3=float((a.topk(3,1).indices==y[:,None]).any(1).float().mean()),
            hollow_accuracy=float((b.argmax(1)==h).float().mean()),per_font={f['family']:float(correct[y==j].float().mean()) for j,f in enumerate(fonts)})
    if args.warmstart:
        best=evaluate('selection')['loss'];torch.save(initial,out/'font_matcher.pt')
    for step in range(initial_step+1,args.steps+1):
        model.train();x,y,h=arrays['train'];ids=rng.integers(len(x),size=64);a,b,_=model(x[ids]);loss=F.cross_entropy(a,y[ids])+.4*F.cross_entropy(b,h[ids])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step%100==0:
            metrics=evaluate('selection');record=dict(step=step,train_loss=float(loss.detach()),selection=metrics);history.append(record)
            write(out/'training.json',history)
            if metrics['loss']<best:
                best=metrics['loss'];torch.save(dict(state_dict=model.state_dict(),step=step,fonts=[f['sha256'] for f in fonts]),out/'font_matcher.pt')
            event(f'Font training {step}/{args.steps}: synthetic selection top-1 {metrics["font_top1"]:.1%}, top-3 {metrics["font_top3"]:.1%}',completed=step,total=args.steps)
    saved=torch.load(out/'font_matcher.pt',weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
    test=evaluate('test');write(out/'synthetic_test.json',dict(selected_step=saved['step'],metrics=test,chance_top1=1/len(fonts),real_accuracy_measured=False))
    frozen=digest(out/'font_matcher.pt');event(f'Font model frozen. Synthetic test top-1 {test["font_top1"]:.1%}; beginning real inference.')
    del arrays,opt
    selections=read(ROOT/'runs/style_phase1_05/results.json')['families']
    family_results={};complete=Image.new('RGB',(1000,1050),'#eef1f5');draw=ImageDraw.Draw(complete)
    for family_index,name in enumerate(DATASETS):
        event(f'Frozen font inference and typography search: {name}')
        measured=read(ROOT/'runs/style_phase1_05'/(name+'_measurements.json'))['fit'];fit=measured[:96];dev=measured[96:]
        oldrun=ROOT/'runs'/('style_frozen_'+name+('_02' if name=='darkbay' else '_01'))
        oldprotocol=read(oldrun/'protocol.json')
        for feature,path in oldprotocol['models'].items():assert digest(path)==oldprotocol['model_sha256_before'][feature]
        preds={r['pixel_sha256']:r for r in (json.loads(line) for line in (oldrun/'predictions.jsonl').read_text().splitlines())}
        for row in measured:
            assert pixel_hash(load(row['path']))==row['pixel_sha256']
            row['style']=copy.deepcopy(row['style']);row['style']['classifier_features']={f:r['image_state'] for f,r in preds[row['pixel_sha256']]['predictions'].items()}
        fp,fh,fe=infer(model,[load(r['path']) for r in fit]);dp,dh,de=infer(model,[load(r['path']) for r in dev])
        posterior=fp.mean(0);top=np.argsort(posterior)[-3:][::-1].tolist();folder=out/name;folder.mkdir()
        write(folder/'real_inference.json',dict(fonts=[f['family'] for f in fonts],fit_probabilities=fp.tolist(),development_probabilities=dp.tolist(),
            fit_hollow_probabilities=fh[:,1].tolist(),development_hollow_probabilities=dh[:,1].tolist(),
            fit_pixel_hashes=[r['pixel_sha256'] for r in fit],development_pixel_hashes=[r['pixel_sha256'] for r in dev],
            top3=[fonts[i]['family'] for i in top],mean_max_confidence=float(fp.max(1).mean()),
            interpretation='nearest candidates within a small font bank; not verified source-font identities'))
        old=selections[name]['selected'];baseline=dict(font=old['font'],effect=None,width=1.,height=1.,baseline=True)
        candidates=[baseline]+[dict(font=fi,effect=effect,width=width,height=height,baseline=False) for fi,effect,width,height in itertools.product(top,['filled','bold','hollow'],[.9,1.1],[1.,1.25])]
        def generate(row,c,seed):
            style=copy.deepcopy(row['style'])
            if not c['baseline']:
                style.update(font_effect=c['effect'],glyph_width_gain=c['width'],text_height=float(np.clip(style['text_height']*c['height'],.25,.75)))
            text=''.join(np.random.default_rng(seed+1000).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5))
            return render(style,text,fonts[c['font']]['path'],seed,old['variant'])
        def score(images,prob,hollow,embedding):
            a,b,z=infer(model,images)
            return float(np.mean((z.mean(0)-embedding.mean(0))**2)+.2*np.mean((a.mean(0)-prob.mean(0))**2)+.05*np.mean((b.mean(0)-hollow.mean(0))**2))
        failures=[]
        for ci,c in enumerate(candidates):
            ims=[]
            for i,row in enumerate(fit[:6]):
                try:ims.append(generate(row,c,61000+i)[0])
                except ValueError as exc:failures.append(dict(stage='search',candidate=ci,index=i,error=str(exc)))
            c['fit_loss']=score(ims,fp,fh,fe)+(6-len(ims)) if ims else 1e6
        eligible=sorted(candidates,key=lambda c:c['fit_loss'])[:4]
        if baseline not in eligible:eligible.append(baseline)
        for c in eligible:
            ims=[]
            for i,row in enumerate(dev[:8]):
                try:ims.append(generate(row,c,62000+i)[0])
                except ValueError as exc:failures.append(dict(stage='development',index=i,error=str(exc)))
            c['development_loss']=score(ims,dp,dh,de)+(8-len(ims)) if ims else 1e6
        selected=min(eligible,key=lambda c:c['development_loss']);write(folder/'selection.json',dict(candidates=candidates,selected=selected))
        selected_images=[];baseline_images=[];recipes=[]
        for i,row in enumerate(dev):
            try:
                im,mask,recipe=generate(row,selected,63000+i);base,_,_=generate(row,baseline,63000+i)
                im.save(folder/f'{i:03d}.png');mask.save(folder/f'{i:03d}_mask.png');base.save(folder/f'{i:03d}_baseline.png')
                selected_images.append(im);baseline_images.append(base)
                recipe.update(reference_profile_hash=row['pixel_sha256'],font_sha256=fonts[selected['font']]['sha256'],phase3_controls=selected);recipes.append(recipe)
            except ValueError as exc:failures.append(dict(stage='generation',index=i,error=str(exc)))
        write(folder/'recipes.json',recipes);write(folder/'failures.json',failures)
        family_results[name]=dict(font=fonts[selected['font']]['family'],effect=selected['effect'],width=selected['width'],height=selected['height'],
            generated=len(selected_images),failures=len(failures),baseline_development_loss=baseline['development_loss'],selected_development_loss=selected['development_loss'],
            mean_font_confidence=float(fp.max(1).mean()),fit_hollow_probability=float(fh[:,1].mean()),real_accuracy_measured=False)
        panel=Image.new('RGB',(1000,480),'#eef1f5');d=ImageDraw.Draw(panel)
        for row,(title,ims) in enumerate([('Real development references',[load(r['path']) for r in dev[:4]]),('Phase 2 recipe baseline',baseline_images[:4]),('Phase 3 font-adjusted candidates',selected_images[:4])]):
            d.text((10,row*160+8),name+' | '+title,fill='black')
            for col,im in enumerate(ims):
                factor=min(230/im.width,110/im.height);view=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.NEAREST);panel.paste(view,(10+col*250,row*160+30))
        panel.save(out/(name+'_comparison.png'))
        y=family_index*210;draw.text((10,y+5),name+' | Phase 2 above; Phase 3 below | '+fonts[selected['font']]['family'],fill='black')
        for ri,ims in enumerate([baseline_images[:4],selected_images[:4]]):
            for col,im in enumerate(ims):
                view=im.copy();view.thumbnail((230,80));complete.paste(view,(10+col*250,y+25+ri*90))
        write(out/'real_development_results.json',family_results)
        for feature,path in oldprotocol['models'].items():assert digest(path)==oldprotocol['model_sha256_before'][feature]
    complete.save(out/'all_families.png');assert digest(out/'font_matcher.pt')==frozen
    write(out/'status.json',dict(state='TRAINED_TESTED_REAL_DEVELOPMENT_REVIEW_PENDING',elapsed_seconds=time.perf_counter()-start,font_model_frozen_sha256=frozen,
        synthetic_test=test,original_feature_models_unchanged=True,real_classifier_training=False,real_heldout_evaluation=False))
    event('Font training, synthetic test and real development inference complete. Visual review pending.',artifacts=[f'runs/{out.name}/all_families.png',f'runs/{out.name}/synthetic_test.json'])
    print(json.dumps(dict(synthetic_test=test,real_development=family_results),indent=2))

if __name__=='__main__':main()
