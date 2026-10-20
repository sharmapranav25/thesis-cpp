"""Portable Phase1 CV+frozen inference -> Phase2 font-aware generation. No training."""
import argparse,copy,csv,hashlib,json,string,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,load,pixel_hash,digest
from style_pipeline.render import render

def verify():
    manifest=read(ROOT/'INTEGRITY.json')
    for relative,expected in manifest.items():
        if digest(ROOT/relative)!=expected:raise ValueError('Release integrity mismatch: '+relative)
    return digest(ROOT/'INTEGRITY.json')

def partition_for(text):
    n=int(hashlib.sha256(('phase12-text-v1:'+text).encode()).hexdigest()[:8],16)%20
    return 'train' if n<18 else ('validation' if n==18 else 'test')

def make_text(rng,bucket,length,partition):
    for attempt in range(10000):
        subtype='mixed'
        if bucket=='numeric':chars=string.digits;subtype='digits'
        elif bucket=='alphabetic':
            subtype=rng.choice(['upper','lower','mixed']);chars={'upper':string.ascii_uppercase,'lower':string.ascii_lowercase,'mixed':string.ascii_letters}[subtype]
        else:chars=string.ascii_letters+string.digits
        text=''.join(rng.choice(list(chars),length))
        if bucket=='alphanumeric' and not(any(c.isalpha() for c in text) and any(c.isdigit() for c in text)):continue
        if bucket=='alphabetic' and subtype=='mixed' and not(any(c.isupper() for c in text) and any(c.islower() for c in text)):continue
        if partition_for(text)==partition:return text,str(subtype)
    raise RuntimeError('Cannot draw text from requested partition')

def generate(args):
    identity=verify();profile=read(args.profile if args.profile else ROOT/'profiles'/f'{args.family}.json');fonts=read(ROOT/'assets/fonts.json')
    if args.count<1:raise ValueError('count must be positive')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    if args.masks:(out/'masks').mkdir()
    cells=[(bucket,n) for bucket in ('numeric','alphabetic','alphanumeric') for n in range(2,9)];rng=np.random.default_rng(args.seed);counts={f'{b}_{n}':0 for b,n in cells};failures=[];seen=set();previews=[]
    write(out/'protocol.json',dict(release_integrity=identity,profile_sha256=digest(args.profile if args.profile else ROOT/'profiles'/f'{args.family}.json'),family=profile['family'],seed=args.seed,partition=args.partition,count=args.count,mode=args.mode,cells=counts,labels='generated, never recognized from reference',count_inference=False,phase3=False,reference_pixel_hashes=[r['reference_pixel_sha256'] for r in profile['entries']],text_partition='stable SHA256 text hash mod20;18train/1validation/1test; duplicate strings within partition allowed',unique_pixels=True))
    with (out/'labels.csv').open('w',newline='',encoding='utf8') as csvfile,(out/'manifest.jsonl').open('w',encoding='utf8') as manifest:
        writer=csv.writer(csvfile);writer.writerow(['filename','label'])
        for index in range(args.count):
            bucket,n=cells[index%21]
            for retry in range(100):
                text,subtype=make_text(rng,bucket,n,args.partition);entry_index=int(rng.integers(len(profile['entries'])));entry=profile['entries'][entry_index];style=copy.deepcopy(entry['style']);font_index=profile['font_index'];seed=int(rng.integers(0,2**31-1))
                if args.mode=='diverse':
                    style['text_height']*=float(rng.uniform(.85,1.15));style['spacing_gain']=float(rng.uniform(.85,1.15))
                    for key in ('line_count_gain','dot_count_gain','curve_count_gain'):style[key]=float(rng.uniform(.7,1.3))
                    if rng.random()<.3:font_index=int(rng.integers(len(fonts)))
                try:
                    image,mask,recipe=render(style,text,str(ROOT/fonts[font_index]['path']),seed,profile['variant']);h=pixel_hash(image)
                    if h in seen:raise ValueError('Duplicate generated pixels')
                    if len(recipe['glyphs'])!=len(text):raise ValueError('Glyph/label mismatch')
                except ValueError as exc:
                    failures.append(dict(index=index,retry=retry,error=str(exc)));continue
                break
            else:raise RuntimeError(f'Failed sample {index}; no silent blank or relabeling')
            seen.add(h);filename=f'images/{index:07d}.png';image.save(out/filename)
            if args.masks:mask.save(out/'masks'/f'{index:07d}.png')
            writer.writerow([filename,text]);record=dict(filename=filename,label=text,bucket=bucket,length=n,alphabetic_subtype=subtype,partition=args.partition,pixel_sha256=h,render_seed=seed,profile_entry=entry_index,reference_pixel_sha256=entry['reference_pixel_sha256'],font_index=font_index,variant=profile['variant'],font_effect=style.get('font_effect'),parameters={k:style.get(k) for k in ('text_height','spacing_gain','line_count_gain','dot_count_gain','curve_count_gain')},minimum_glyph_visibility=min(g['visible_fraction'] for g in recipe['glyphs']))
            if args.recipes:record['recipe']=recipe
            manifest.write(json.dumps(record)+'\n');counts[f'{bucket}_{n}']+=1
            if index<42:previews.append((image.copy(),f'{bucket} {n}: {text}'))
            if (index+1)%1000==0:print('generated',index+1,flush=True);write(out/'status.json',dict(state='running',generated=index+1))
    gallery=Image.new('RGB',(1000,110*((len(previews)+3)//4)),'white');draw=ImageDraw.Draw(gallery)
    for i,(im,label) in enumerate(previews):
        im.thumbnail((235,80));x=10+(i%4)*250;y=(i//4)*110;draw.text((x,y+3),label,fill='black');gallery.paste(im,(x,y+22))
    gallery.save(out/'preview.png');write(out/'failures.json',failures);write(out/'status.json',dict(state='complete',generated=args.count,cells=counts,retries=len(failures)));print('COMPLETE',out)

def profile(args):
    verify()
    import torch,joblib
    from feature_bank.descriptors import extract
    from feature_bank.regions import regions
    from style_pipeline.measure import extract_style
    from style_pipeline.font_match import FontMatcher,infer
    spec=read(args.references)
    if spec.get('partition')!='style_fit':raise ValueError('Reference manifest must explicitly declare partition=style_fit')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);thresholds=read(ROOT/'assets/thresholds.json');models={f:joblib.load(ROOT/'assets'/f'{f}_LR.joblib') for f in thresholds};entries=[];images=[];seen=set()
    for raw in spec['images']:
        path=Path(raw);path=path if path.is_absolute() else Path(args.references).resolve().parent/path;im=load(path);h=pixel_hash(im)
        if h in seen:continue
        seen.add(h);images.append(im);blocks=regions(*im.size,scales=((3,10),));xx=np.stack([extract(im.crop(r['box']))[0] for r in blocks]);predictions={}
        for feature,bundle in models.items():
            raw=bundle['model'].decision_function(xx);probs=bundle['calibrator'].predict_proba(raw[:,None])[:,1];lo,hi=thresholds[feature]['thresholds'];states=['present' if p>=hi else 'absent' if p<=lo else 'uncertain' for p in probs];state='present' if 'present' in states else 'absent' if all(s=='absent' for s in states) else 'uncertain';predictions[feature]=dict(image_state=state,block_probabilities=probs.tolist(),block_states=states)
        style=extract_style(im);style['classifier_features']={k:v['image_state'] for k,v in predictions.items()};entries.append(dict(style=style,reference_pixel_sha256=h,reference_path_provenance=str(path),feature_predictions=predictions))
    if not entries:raise ValueError('No reference images')
    fonts=read(ROOT/'assets/fonts.json');device=args.device;model=FontMatcher(len(fonts)).to(device);model.load_state_dict(torch.load(ROOT/'assets/font_matcher.pt',map_location=device,weights_only=True)['state_dict']);model.eval();p,h,_=infer(model,images,device=device);chosen=int(p.mean(0).argmax());effect='hollow' if h[:,1].mean()>=.7 else 'filled'
    for entry in entries:entry['style']['font_effect']=effect
    write(out/'profile.json',dict(schema=1,family=args.name,variant=args.variant,font_index=chosen,font_effect=effect,font_probabilities=p.mean(0).tolist(),entries=entries,source='fresh CV+frozen feature and font inference on explicitly declared style_fit images',real_labels_used=False));print('PROFILE',out/'profile.json')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('verify');sub.add_parser('list')
    g=sub.add_parser('generate');source=g.add_mutually_exclusive_group(required=True);source.add_argument('--family');source.add_argument('--profile');g.add_argument('--out',required=True);g.add_argument('--count',type=int,default=210);g.add_argument('--seed',type=int,required=True);g.add_argument('--partition',choices=['train','validation','test'],required=True);g.add_argument('--mode',choices=['baseline','diverse'],default='baseline');g.add_argument('--masks',action='store_true');g.add_argument('--recipes',action='store_true')
    q=sub.add_parser('profile');q.add_argument('--references',required=True);q.add_argument('--out',required=True);q.add_argument('--name',required=True);q.add_argument('--variant',type=int,choices=[0,1,2,3],default=0);q.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.command=='verify':print('VERIFIED',verify())
    elif args.command=='list':print(json.dumps(read(ROOT/'profiles/index.json'),indent=2))
    elif args.command=='generate':generate(args)
    else:profile(args)
if __name__=='__main__':main()
