from core8 import *
from concurrent.futures import ProcessPoolExecutor
import time,json,argparse
from PIL import ImageDraw,ImageFont,ImageOps

def gallery(family,refs,prior_rows,rows):
    canvas=Image.new('RGB',(900,690),'#f4f6f8');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    d.text((12,8),family+' | Meta-Sim2 adaptation',fill='#182033',font=font)
    for j,name in enumerate(['Original','Generic policy prior','Trained policy']):d.text((12+300*j,45),name,fill='#182033',font=font)
    for i in range(6):
        for j,im in enumerate([Image.open(refs[i]['source']).convert('RGB'),Image.fromarray(prior_rows[i]['image']),Image.fromarray(rows[i]['image'])]):
            tile=ImageOps.contain(im,(286,78),Image.Resampling.NEAREST);canvas.paste(tile,(300*j+(300-tile.width)//2,82+i*96+(78-tile.height)//2))
    d.text((12,665),'First 6 draws, no cherry-picking. Synthetic strings are independently generated.',font=small,fill='#465364')
    canvas.save(ROOT/family/'comparison.png')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=80);parser.add_argument('--family',default=None);args=parser.parse_args()
    started=time.time();torch.manual_seed(81001);np.random.seed(81001)
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');g=grammar(fonts);U.dump(ROOT/'grammar.json',g)
    model=Features();policy=Policy(g);opt=torch.optim.Adam(policy.parameters(),lr=.003);rng=np.random.default_rng(81002)
    # Maximum likelihood pretraining on the generic, target-independent structure prior.
    for _ in range(250):
        a=prior(g,128,rng);_,logp,_=policy(128,a);loss=-logp.mean();opt.zero_grad();loss.backward();opt.step()
    torch.save(policy.state_dict(),ROOT/'prior.pt');initial=copy.deepcopy(policy.state_dict())
    U.dump(ROOT/'protocol.json',dict(method='Meta-Sim2 paper-based adaptation, not official code',paper='https://arxiv.org/abs/2008.09092',
        policy='Autoregressive GRU, masked grammar productions; up to three distinct effects with STOP; one fixed learned latent/start state.',
        objective='Per-sample Gaussian KDE log q - log p; REINFORCE with moving-average baseline; synthetic validity penalty; entropy bonus 0.002.',
        differences='CAPTCHA grammar and discretized parameters; ImageNet ResNet18 plus pixel/edge features with weighted standardization; bounded effect grammar guarantees termination; no separate continuous-parameter network.',
        initialization='250 MLE steps from generic structure prior. No hybrid predictions and no earlier controller outputs used in training.',
        target_access='RGB pixels and image dimensions only. Fit colors from unsupervised quantization. No filenames parsed, text, masks, boxes, OCR or target metadata.',
        validation='Every 10 updates, fixed 48 synthetic seeds and policy sampling seed; lowest selection KDE plus validity penalty including step0. Test never selects checkpoint.',
        diagnostics='Previously exposed diagnostic target split. Fresh 48 synthetic draws per arm; not unseen-target or solver-generalization evidence.',
        steps=args.steps,batch=24,seed=81001,feature_weights_sha256=U.sha(ROOT/'resnet18-f37072fd.pth'),
        source_sha256={p.name:U.sha(p) for p in [ROOT/'core8.py',ROOT/'build8.py']}))
    summary=U.read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {}
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family in ([args.family] if args.family else FAMILIES):
            if family in summary:continue
            dest=ROOT/family;dest.mkdir(exist_ok=True);split=U.read(WORK/'controller_mvp_v7_dino'/family/'target_splits.json')
            sets=[{r['pixel_sha256'] for r in split[k]} for k in ['fit','selection','test']]
            assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
            U.dump(dest/'target_splits.json',split)
            fit=load_images(split['fit']);val=load_images(split['selection']);size=np.median([[x.shape[1],x.shape[0]] for x in fit],axis=0).astype(int).tolist()
            colors=pixel_palette(fit);U.dump(dest/'colors.json',colors)
            policy.load_state_dict(initial);opt=torch.optim.Adam(policy.parameters(),lr=.0015)
            def draw(n,seed):
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    with torch.no_grad():a,_,_=policy(n)
                return [decode(g,x,colors) for x in a.numpy()]
            def recipes(n,offset):return random_recipes(81117,n,size,fonts,offset)
            vr=recipes(48,100000);cal=render_batch(pool,recipes(48,200000),draw(48,81222))
            features_fit=model.extract(fit);density=Density(features_fit,model.extract([x['image'] for x in cal]));tf=density.transform(features_fit);vf=density.transform(model.extract(val))
            U.dump(dest/'feature_normalization.json',dict(mean=density.mean.tolist(),std=density.std.tolist(),weights=density.weights.tolist()))
            def evaluate(rec,seed,target):
                rows=render_batch(pool,rec,draw(len(rec),seed));features=density.transform(model.extract([r['image'] for r in rows]))
                c=density.cost(features,target)
                return dict(objective=float((c+penalty(rows)).mean()),kde=float(c.mean()),complete=float(np.mean([r['complete'] for r in rows])),usable=float(np.mean([r['usable'] for r in rows]))),rows
            best,unused=evaluate(vr,81333,vf);best_step=0;torch.save(policy.state_dict(),dest/'selected.pt');history=[dict(step=0,validation=best)]
            baseline=None
            for step in range(1,args.steps+1):
                torch.manual_seed(82000+step);a,logp,entropy=policy(24)
                rows=render_batch(pool,recipes(24,step*24),[decode(g,x,colors) for x in a.detach().numpy()])
                q=density.transform(model.extract([r['image'] for r in rows]));cost=density.cost(q,tf)+penalty(rows)
                if baseline is None:baseline=float(cost.mean())
                advantage=torch.tensor(cost-baseline);loss=(advantage*logp).mean()-.002*entropy.mean()
                opt.zero_grad();loss.backward();grad=float(nn.utils.clip_grad_norm_(policy.parameters(),5));opt.step();baseline=.9*baseline+.1*float(cost.mean())
                record=dict(step=step,cost=float(cost.mean()),loss=float(loss),entropy=float(entropy.mean()),grad=grad)
                if step%10==0 or step==args.steps:
                    score,_=evaluate(vr,81333,vf);record['validation']=score
                    if score['objective']<best['objective']:
                        best=score;best_step=step;torch.save(policy.state_dict(),dest/'selected.pt')
                    U.dump(ROOT/'status.json',dict(family=family,step=step,selected_step=best_step,seconds=time.time()-started))
                    print(family,step,'validation',round(score['objective'],4),'best',best_step,flush=True)
                history.append(record)
            torch.save(policy.state_dict(),dest/'last.pt');U.dump(dest/'training.json',history)
            # Only after checkpoint freeze, load diagnostic pixels and evaluate fresh draws.
            test=density.transform(model.extract(load_images(split['test'])));tr=recipes(48,300000)
            policy.load_state_dict(initial);prior_score,prior_rows=evaluate(tr,81444,test)
            policy.load_state_dict(torch.load(dest/'selected.pt',weights_only=True));score,rows=evaluate(tr,81444,test)
            save_rows(dest,'prior',prior_rows);save_rows(dest,'trained',rows);gallery(family,split['test'],prior_rows,rows)
            controls=draw(48,81444);U.dump(dest/'sampled_controls.json',controls)
            summary[family]=dict(selected_step=best_step,validation=best,prior=prior_score,trained=score,
                unique_recipes=len({json.dumps(x,sort_keys=True) for x in controls}),policy_sha256=U.sha(dest/'selected.pt'),
                parameter_change=float(sum((v-initial[k]).square().sum() for k,v in policy.state_dict().items())))
            U.dump(ROOT/'summary.json',summary);print(family,summary[family],flush=True)
    U.dump(ROOT/'completion.json',dict(seconds=time.time()-started,families=list(summary),complete=all(x in summary for x in FAMILIES)))

if __name__=='__main__':main()
