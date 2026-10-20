"""Matched black-box controller MVPs, selected on separate pixel batches."""
from features9 import *
from run import load_images,save_rows
from build7 import fresh
import render7 as R
from concurrent.futures import ProcessPoolExecutor
from PIL import ImageDraw,ImageFont,ImageOps
import argparse,copy,time

def palette(ims,masks):
    arr=np.stack([pixels(x) for x in ims])
    def quant(w):
        data=arr[w][::3]
        if len(data)<8:data=arr.reshape(-1,3)[::3]
        q=Image.fromarray(data.reshape(1,-1,3)).quantize(colors=6);pal=q.getpalette()
        return [pal[3*i:3*i+3] for _,i in sorted(q.getcolors(),reverse=True)]
    return dict(text=quant(masks>.5),background=quant(masks<.2))

def render(pool,recipes,p):
    rows=list(pool.map(R.safe_render,[(r,p) for r in recipes]))
    for row,r in zip(rows,recipes):
        if 'image' not in row:
            w,h=r['size'];row.update(image=np.full((h,w,3),255,np.uint8),mask=np.zeros((h,w),np.uint8),seed=r['seed'],text='RENDER FAILURE')
    return rows

def panel(path,title,columns,rows):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    sheet=Image.new('RGB',(300*len(columns),90+94*len(rows)+28),'#f4f6f8');d=ImageDraw.Draw(sheet)
    d.text((10,8),title,font=font,fill='#182033')
    for j,c in enumerate(columns):d.text((10+j*300,46),c,font=font,fill='#182033')
    for i,row in enumerate(rows):
        for j,im in enumerate(row):
            tile=ImageOps.contain(Image.fromarray(im),(284,80),Image.Resampling.NEAREST)
            sheet.paste(tile,(300*j+(300-tile.width)//2,84+i*94+(80-tile.height)//2))
    d.text((10,sheet.height-24),'First samples; target strings are not supplied to the controller.',font=small,fill='#465364');sheet.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['sam','dinov3'],required=True);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--family',choices=FAMILIES);args=ap.parse_args()
    root=ROOT/args.backend;root.mkdir(exist_ok=True)
    model=(SAM if args.backend=='sam' else DINOv3)(args.checkpoint)
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');started=time.time();summary=U.read(root/'summary.json') if (root/'summary.json').exists() else {}
    U.dump(root/'protocol.json',dict(backend=args.backend,weights_sha256=model.weight_sha,vlm=False,target_answers=False,
        renderer='Unchanged MVP7 renderer, all fonts and effects generic.',initialization='Same frozen hybrid recipe bank for both backends; no earlier optimized controller profile used.',
        perception='Frozen official backbone. SAM: automatic point-grid proposals plus generic geometry filter; DINOv3: dense patch features and two-cluster border-background heuristic. Neither mask is ground truth.',
        fit='First24 fit target pixels, 3 fresh synthetic strings per candidate, sequential coordinate search.',
        validation='First24 selection target pixels; best4 fit candidates plus neutral baseline; 12 fresh synthetic samples each.',
        test='First24 previously exposed diagnostic pixels; 24 fresh renders per arm. No test selection.',
        objective='Weighted feature-distribution energy distance. Appearance is reported separately from validity penalties.',
        caveat='Small MVP search budget. Masks may include frames/background; no verified target text segmentation or solver transfer result.',
        code_sha256={n:U.sha(ROOT/n) for n in ['features9.py','build9.py']}))
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family in ([args.family] if args.family else FAMILIES):
            if family in summary:continue
            dest=root/family;dest.mkdir(exist_ok=True);old=WORK/'controller_mvp_v7_dino'/family
            split=U.read(old/'target_splits.json');split={k:v[:24] for k,v in split.items()}
            sets=[{x['pixel_sha256'] for x in split[k]} for k in ['fit','selection','test']];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
            U.dump(dest/'target_splits.json',split);fit=load_images(split['fit']);target=model.extract(fit);val=model.extract(load_images(split['selection']))
            colors=palette(fit,target['mask']);U.dump(dest/'palettes.json',colors)
            previews=[]
            for im,m in zip(fit[:6],target['mask'][:6]):
                im=pixels(im);overlay=im.copy();overlay[m>.5]=(overlay[m>.5]*.45+np.array([0,220,120])*.55).astype(np.uint8)
                previews.append([im,overlay,np.repeat((m[:,:,None]*255).astype(np.uint8),3,2)])
            panel(dest/'regions.png',family+' | '+args.backend+' inferred regions',['Target','Region overlay','Inferred mask'],previews)
            bank=U.read(old/'recipe_banks.json')['hybrid'];recipes=dict(fit=fresh(bank['fit'],3,91000),selection=fresh(bank['validation'][0],12,92000),test=fresh(bank['test'],24,93000))
            U.dump(dest/'recipe_bank.json',recipes);neutral=R.expand({});baseline=render(pool,recipes['fit'],neutral);obj=Objective(target,model.extract([r['image'] for r in baseline]));records=[];seen=set()
            U.dump(dest/'normalization.json',dict(std={k:v.tolist() for k,v in obj.std.items()},scale=obj.scale))
            def evaluate(p,phase):
                p=R.expand(p);key=json.dumps(p,sort_keys=True)
                if key in seen:return
                seen.add(key);rows=render(pool,recipes['fit'],p);f=model.extract([r['image'] for r in rows]);score=obj.score(f,target,rows)
                records.append(dict(index=len(records),phase=phase,controls=p,score=score))
                U.dump(root/'status.json',dict(family=family,phase=phase,candidates=len(records),seconds=time.time()-started))
            def leader():return min(records,key=lambda r:r['score']['objective'])['controls']
            def group(name,options):
                for p in options:evaluate(p,name)
                print(args.backend,family,name,len(records),round(min(r['score']['objective'] for r in records),3),flush=True)
            evaluate(neutral,'neutral');p=leader();clean={x:0 for x in R.C.R.R.SAMPLED}
            group('palette',[{**p,'background':colors['background'][0],'text_palette':colors['text'],'effect_palette':colors['text'],'color_mode':mode,'effects':effects} for mode in ['single','alternating','random'] for effects in [p['effects'],clean]])
            p=leader();group('background',[{**p,'background':c} for c in colors['background'][:3]])
            p=leader();idx=np.random.default_rng(91031).choice(len(fonts),8,replace=False);group('font',[{**p,'font':fonts[i]['key']} for i in idx])
            fields=dict(length=[3,4,5,6,7,8],charset=['digits','upper','lower','upper_digits','mixed'],size_absolute=[.45,.65,.85],
                width_absolute=[.7,1.,1.2],rotation_absolute=[0.,15.,35.],layout=['line','curve'],spacing_absolute=[-.15,0.,.2])
            for field,values in fields.items():
                p=leader();group(field,[{**p,field:v} for v in values])
            p=leader();group('decoration',[{**p,'glyph_outline':outline,'frame_width':frame} for outline,frame in [(0,0),(1,0),(2,0),(0,1),(1,1)]])
            for effect in ['gradient','texture','blocks','grid','lines','bands','dots','pattern','hollow','wave']:
                p=leader();group('effect_'+effect,[{**p,'effects':{**p['effects'],effect:level}} for level in [0,3]])
            U.dump(dest/'fit_history.json',records)
            candidates=sorted(records,key=lambda r:r['score']['objective'])[:4]
            if records[0] not in candidates:candidates.append(records[0])
            selection=[]
            for r in candidates:
                rows=render(pool,recipes['selection'],r['controls']);s=obj.score(model.extract([x['image'] for x in rows]),val,rows)
                selection.append(dict(index=r['index'],controls=r['controls'],score=s))
            chosen=min(selection,key=lambda x:x['score']['objective']);U.dump(dest/'selection.json',selection);U.dump(dest/'selected.json',chosen);digest=U.sha(dest/'selected.json')
            # Freeze before reading the diagnostic split.
            test=load_images(split['test']);tf=model.extract(test);scores={};rendered={}
            for name,p in [('baseline',neutral),('controller',chosen['controls'])]:
                rows=render(pool,recipes['test'],p);f=model.extract([r['image'] for r in rows]);scores[name]=obj.score(f,tf,rows)
                truth=np.stack([np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>127 for r in rows]);pred=f['mask']>.5
                scores[name]['synthetic_mask_iou']=float(np.mean((pred&truth).sum((1,2))/np.maximum(1,(pred|truth).sum((1,2)))))
                save_rows(dest,name,rows);rendered[name]=rows
            assert U.sha(dest/'selected.json')==digest
            panel(dest/'renders.png',family+' | '+args.backend+' controller',['Original','Hybrid baseline',args.backend+' controller'],[[pixels(test[i]),rendered['baseline'][i]['image'],rendered['controller'][i]['image']] for i in range(6)])
            result=dict(scores=scores,candidates=len(records),selected_index=chosen['index'],selected_sha256=digest,target_mask_coverage=float(target['mask'].mean()),weight_sha256=model.weight_sha)
            U.dump(dest/'results.json',result);summary[family]=result;U.dump(root/'summary.json',summary)
            print(args.backend,family,'DONE', {k:round(s['appearance'],3) for k,s in scores.items()},flush=True)
            model.cache.clear()
    U.dump(root/'completion.json',dict(families=list(summary),complete=all(f in summary for f in FAMILIES),seconds=time.time()-started))

if __name__=='__main__':main()
