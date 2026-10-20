from pathlib import Path
import json,shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image,ImageDraw
import run_v2_tier1_04 as v4
base=v4.base;HERE=Path(__file__).parent;OUT=HERE/'v2_curve_debug_01'

def main():
    OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/Path(__file__).name);torch.set_num_threads(4)
    fonts=json.loads((HERE/'v2_tier1_04/fonts.json').read_text())
    items=[base.generate(fonts[:6],51000000+i) for i in range(420)]
    x=torch.tensor(np.stack([r[0] for r in items]).transpose(0,3,1,2)/255,dtype=torch.float32)
    y=torch.tensor(np.stack([r[2] for r in items]));m=torch.tensor(np.stack([r[1] for r in items]));local=v4.local_targets(m)
    result={'protocol':'Post-hoc debugging on previously evaluated51m seed benchmark. No fresh accuracy claim. Frozen inference ablations are sensitivity probes, not retraining comparisons.'}
    preds={};heads={};weights={}
    for name,separate in [('original03',None),('shared_local_head',False),('grid_specialists',True)]:
        model=(base.Multi(True) if separate is None else v4.Specialists(separate)).cuda().eval()
        file=HERE/'v2_tier1_03/multiview.pt' if separate is None else HERE/'v2_tier1_04'/(name+'.pt')
        model.load_state_dict(torch.load(file,weights_only=True)['state_dict'])
        def infer():
            aa=[];hh=[];ww=[]
            with torch.no_grad():
                for i in range(0,len(x),12):
                    a,z=model(x[i:i+12].cuda());aa.append(a.sigmoid().cpu())
                    if separate is not None:hh.append(model.last_local.sigmoid().cpu());ww.append(model.last_grid_weights.cpu())
            return torch.cat(aa),torch.cat(hh) if hh else None,torch.cat(ww) if ww else None
        p,h,w=infer();preds[name]=p;heads[name]=h;weights[name]=w
        t=y[:,1].bool();fp=(p[:,1]>=.5)&~t
        r=dict(curve=v4.metrics(p,y),curve_auc=roc_auc_score(t,p[:,1]),curve_average_precision=average_precision_score(t,p[:,1]),false_positive_count=int(fp.sum()),false_positive_seeds=[items[i][4]['seed'] for i in torch.where(fp)[0]],negative_subgroups={})
        for label,select in [('all_negatives',~t),('clean_text_only',y.sum(1)==0),('straight_without_curve',(y[:,0]>0)&~t),('dots_without_curve',(y[:,2]>0)&~t),('blocks_without_curve',(y[:,3]>0)&~t),('grain_without_curve',(y[:,4]>0)&~t)]:
            r['negative_subgroups'][label]=dict(n=int(select.sum()),fpr=float((p[select,1]>=.5).float().mean()),mean_curve_score=float(p[select,1].mean()))
        if separate is not None:
            r['grids']={};offset=0
            for g,(cols,rows) in enumerate(v4.GRIDS):
                n=cols*rows;hp=h[:,offset:offset+n,1].flatten();ht=local[:,offset:offset+n,1].flatten()
                r['grids'][f'{cols}x{rows}']=dict(auc=roc_auc_score(ht,hp),positive_fraction=float(ht.mean()),mean_attention=float(w[:,g].mean()),curve_f1=v4.metrics(h[:,offset:offset+n].flatten(0,1),local[:,offset:offset+n].flatten(0,1))['f1'][1]);offset+=n
        if separate is True:
            r['frozen_ablation']={}
            for mode in ('neutral_curve_head_scores','coarse_grids_only','fine_grids_only'):
                hooks=[]
                if mode=='neutral_curve_head_scores':
                    def neutral(module,args,out):
                        out=out.clone();out[:,:,1]=0;return out
                    hooks=[head.register_forward_hook(neutral) for head in model.heads]
                else:
                    def restrict(module,args,out):
                        out=out.clone()
                        if mode=='coarse_grids_only':out[:,6:]=-1e4
                        else:out[:,:6]=-1e4
                        return out
                    hooks=[model.grid_score.register_forward_hook(restrict)]
                q,_,_=infer()
                for hook in hooks:hook.remove()
                r['frozen_ablation'][mode]=dict(curve_f1=v4.metrics(q,y)['f1'][1],curve_fpr=v4.metrics(q,y)['fpr'][1],mean_absolute_curve_score_change=float((q[:,1]-p[:,1]).abs().mean()))
        result[name]=r;print(name,r['curve_auc'],r['negative_subgroups'],flush=True)
        del model
    geometry={}
    for cols,rows in v4.GRIDS:
        eligible=nearlystraight=insufficient=0
        for im,visible,labels,full,record in items:
            if record['parameters'][1]['count']!=1 or labels[1]==0:continue
            for j in range(rows):
                for i in range(cols):
                    x0=int(np.ceil(i/cols*127));x1=int(np.floor((i+1)/cols*127))+1;y0=int(np.ceil(j/rows*63));y1=int(np.floor((j+1)/rows*63))+1
                    if visible[1,y0:y1,x0:x1].sum()<3:continue
                    yy,xx=np.where(full[1,y0:y1,x0:x1]>.5);ux=np.unique(xx)
                    if len(ux)<4:insufficient+=1;continue
                    mid=np.array([np.median(yy[xx==a]) for a in ux]);fit=np.polyval(np.polyfit(ux,mid,1),ux);eligible+=1;nearlystraight+=int(np.max(np.abs(mid-fit))<=1.)
        geometry[f'{cols}x{rows}']=dict(eligible_single_curve_positive_crops=eligible,nearly_straight_within_one_native_pixel=nearlystraight,insufficient_x_extent=insufficient,limitation='full-mask centerline residual, single generated sine only; geometric ambiguity proxy, not perceptual ground truth')
    result['geometry']=geometry
    # Fixed error-example selection: first12 specialist false positives.
    bad=torch.where((preds['grid_specialists'][:,1]>=.5)&(y[:,1]==0))[0][:12]
    canvas=Image.new('RGB',(800,420),'white');draw=ImageDraw.Draw(canvas)
    for j,idx in enumerate(bad):
        i=int(idx);cx=(j%4)*200;cy=(j//4)*140;canvas.paste(Image.fromarray(items[i][0]),(cx,cy+20));draw.text((cx,cy),str(items[i][4]['seed']),fill='black');draw.text((cx,cy+87),f"old {preds['original03'][i,1]:.2f} new {preds['grid_specialists'][i,1]:.2f}",fill='black');draw.text((cx,cy+105),'L D R G '+str(y[i,[0,2,3,4]].int().tolist()),fill='black')
    canvas.save(OUT/'curve_false_positives.png');base.write(OUT/'diagnostics.json',result)
    print('DIAGNOSTICS COMPLETE',flush=True)

if __name__=='__main__':main()
