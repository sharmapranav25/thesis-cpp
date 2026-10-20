"""Frozen diagnostics and a visual comparison; never feeds checkpoint selection."""
from core8 import *
from dino5 import DinoComparator,Objective
from PIL import ImageDraw,ImageFont,ImageOps
import json,hashlib

def main():
    summary=U.read(ROOT/'summary.json');dino=DinoComparator();review={};audit={}
    for family,result in summary.items():
        dest=ROOT/family;split=U.read(dest/'target_splits.json');real=dino.extract(load_images(split['test']))
        labels={name:U.read(dest/name/'labels.json') for name in ['prior','trained']}
        images={name:[np.asarray(Image.open(dest/r['image']).convert('RGB')) for r in rows] for name,rows in labels.items()}
        feats={name:dino.extract(ims) for name,ims in images.items()};obj=Objective(real,feats['prior'])
        scores={name:obj.score(feats[name],real,labels[name]) for name in labels}
        review[family]=dict(dino_diagnostic=scores,appearance_improved=result['trained']['kde']<result['prior']['kde'],
            dino_appearance_improved=scores['trained']['appearance']<scores['prior']['appearance'])
        for name in labels:
            for row,im in zip(labels[name],images[name]):
                assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
        history=U.read(dest/'training.json');valid=[r for r in history if 'validation' in r];winner=min(valid,key=lambda r:r['validation']['objective'])
        assert winner['step']==result['selected_step']
        assert result['policy_sha256']==U.sha(dest/'selected.pt')
        assert all(np.isfinite(r['loss']) and np.isfinite(r['grad']) for r in history if 'loss' in r)
        p=Policy(U.read(ROOT/'grammar.json'));p.load_state_dict(torch.load(dest/'selected.pt',weights_only=True));p.eval()
        torch.manual_seed(89123)
        with torch.no_grad():aa,_,entropy=p(512)
        histograms={name:np.bincount(aa[:,i].numpy(),minlength=len(values)).tolist() for i,(name,values) in enumerate(p.g['fields'].items())}
        review[family]['policy_samples']=dict(samples=512,mean_entropy=float(entropy.mean()),rule_counts=histograms)
        initial=torch.load(ROOT/'prior.pt',weights_only=True)
        changed=float(sum((v-initial[k]).square().sum() for k,v in p.state_dict().items()))
        assert abs(changed-result['parameter_change'])<1e-5
        audit[family]=dict(validation_selection_verified=True,all_96_pixels_verified=True,finite_training=True,
            unique_trained_pixels=len({r['pixel_sha256'] for r in labels['trained']}),
            learned_parameters_changed=changed>0,parameter_change=changed)
        sheet=Image.new('RGB',(1200,690),'#f4f6f8');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        draw.text((12,8),family+' | three controllers',font=font,fill='#182033')
        for j,title in enumerate(['Original','Our controller','DINO controller v2','Meta-Sim2 adaptation']):draw.text((12+300*j,45),title,font=font,fill='#182033')
        old=WORK/'controller_mvp_v7_dino';custom=U.read(old/'controller_comparison'/family/'custom/labels.json');dl=U.read(old/family/'dino_v2/labels.json')
        for i in range(6):
            ims=[Image.open(split['test'][i]['source']).convert('RGB'),Image.open(old/'controller_comparison'/family/custom[i]['image']).convert('RGB'),Image.open(old/family/dl[i]['image']).convert('RGB'),Image.fromarray(images['trained'][i])]
            for j,im in enumerate(ims):
                tile=ImageOps.contain(im,(286,78),Image.Resampling.NEAREST);sheet.paste(tile,(300*j+(300-tile.width)//2,82+i*96+(78-tile.height)//2))
        draw.text((12,665),'First 6 draws. Meta-Sim2 uses a generic prior; other controllers use hybrid recipes. Budgets differ.',font=small,fill='#465364');sheet.save(dest/'all_controllers.png')
    U.dump(ROOT/'review.json',review);U.dump(ROOT/'audit.json',audit)
    html=['<!doctype html><meta charset="utf-8"><title>Controller 3: Meta-Sim2 adaptation</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:15px;background:#f4f6f8;color:#182033}img{width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}</style>',
        '<h1>Controller 3: Meta-Sim2 adaptation</h1><p>A trained autoregressive GRU samples renderer recipes. Gaussian feature-density rewards train it through REINFORCE. Frozen ResNet18 and pixel/edge features; no DINO in training, hybrid initialization, VLM, OCR or target labels.</p>',
        '<p>Paper-based adaptation, not official Meta-Sim2 code. All examples are first draws. Different search budgets and initialization make this a practical comparison, not an isolated algorithm benchmark. These target diagnostics were previously seen in earlier experiments.</p>',
        '<h2>Prior versus learned policy</h2><p>Lower is better. KDE appearance excludes the synthetic usability penalty. The DINO appearance check is diagnostic only; it never selected a checkpoint. No solver transfer claim.</p>',
        '<table><tr><th>Family</th><th>Selected step</th><th>KDE appearance</th><th>Usable</th><th>DINO appearance</th></tr>']
    for f,r in summary.items():
        d=review[f]['dino_diagnostic'];html.append(f'<tr><td>{f}</td><td>{r["selected_step"]}</td><td>{r["prior"]["kde"]:.3f} → {r["trained"]["kde"]:.3f}</td><td>{r["prior"]["usable"]:.0%} → {r["trained"]["usable"]:.0%}</td><td>{d["prior"]["appearance"]:.3f} → {d["trained"]["appearance"]:.3f}</td></tr>')
    html.append('</table>')
    for family in summary:html.extend([f'<h2>{family}</h2>',f'<img src="{family}/comparison.png">',f'<img src="{family}/all_controllers.png">'])
    html.append('<p>References: <a href="https://arxiv.org/abs/2008.09092">Meta-Sim2 paper</a>; <a href="https://github.com/nv-tlabs/meta-sim-structure">official repository (code listed as coming soon)</a>.</p>')
    (ROOT/'report.html').write_text('\n'.join(html),encoding='utf8');print(json.dumps(review,indent=2))

if __name__=='__main__':main()
