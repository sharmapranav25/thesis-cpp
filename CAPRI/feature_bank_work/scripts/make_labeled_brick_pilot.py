import argparse
import itertools
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest,load,pixel_hash,new_run,FEATURES as VOCABULARY
from feature_bank.regions import regions,verify_crops,check_group_partitions
from feature_bank.labeled_bricks import FEATURES,POLICY,sample_operations,compose,evidence,crop_labels

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--project',default='E:/pranav/Final sem/sourcefree_synth');parser.add_argument('--out',default='runs/feature_bank_labeled_bricks_01');args=parser.parse_args()
project=Path(args.project);out=new_run(ROOT/args.out)
sys.path.insert(0,str(project))
from sourcefree_synth.renderer import default_parameters,render_once
fonts=read(project/'runs/feature_bank_pilot_01/font_selection.json')[:2]
for f in fonts:
    if digest(f['path'])!=f['sha256']: raise ValueError('Font changed')
write(out/'protocol.json',dict(policy=POLICY,base_groups=8,variants_per_group=8,grid=[3,10],
  renderer='unchanged sourcefree_synth.renderer.render_once for clean text; new instrumented Pillow overlays',
  masks='final image minus leave-one-effect-out replay; not inferred from real pixels',
  role='synthetic label-validation pilot only; NOT an accepted training corpus',
  target_data_access='none',split='all pilot_only; base_group retained for future grouped splitting',
  text_visibility_minimum=0.55,training_eligible=False))
start=time.perf_counter(); images=[];bricks=[];failure=[]
strings=['Il1','OO0Q','ijIl','x8m4K','2Wd7pQ','r5a9','yT6b3','p4E8z2N']
for group,text in enumerate(strings):
    gid=f'base-{group:02d}';folder=out/gid;folder.mkdir();dark=bool(group%2)
    bg=(28,32,38) if dark else (242,242,242);fg=(225,225,225) if dark else (25,30,45)
    profile=dict(id='fixed-generic-brick-pilot',width=240,height=96,background=[list(bg)],foreground=[list(fg)])
    p=default_parameters(0);p.update(background='flat',dots=0,lines=0,rotation=0,jitter=0,outline=False,shadow=False,
        multicolor=False,wave_x=0,wave_y=0,blur=0,height=0.55,width=0.78)
    # Each font appears in both polarities, instead of font identifying polarity.
    base,mask,glyphs,meta=render_once(text,p,profile,[fonts[(group//2)%2]],1700+group)
    base.save(folder/'clean.png');mask.save(folder/'text_mask.png')
    ops=sample_operations(*base.size,seed=3500+group,dark=dark)
    write(folder/'recipe.json',dict(base_metadata=meta,operations=ops,operation_order=FEATURES,
                                   base_group=gid,profile=profile))
    for bits in itertools.product((0,1),repeat=3):
        enabled=[f for f,b in zip(FEATURES,bits) if b];vid=''.join(map(str,bits));vfolder=folder/vid;vfolder.mkdir()
        final,ev=evidence(base,ops,enabled,background=Image.new('RGB',base.size,bg));final.save(vfolder/'image.png')
        if pixel_hash(final)!=pixel_hash(compose(base,ops,enabled)):raise ValueError('Recorded operation replay mismatch')
        background_final=compose(Image.new('RGB',base.size,bg),ops,enabled)
        text_delta=np.max(np.abs(np.asarray(final).astype('int16')-np.asarray(background_final).astype('int16')),axis=2)
        retention=[float((text_delta[np.asarray(g)>100]>=12).mean()) for g in glyphs]
        accepted=min(retention)>=0.55
        if not accepted:failure.append(dict(id=gid+'/'+vid,reason='text contribution below fixed visibility gate',retention=retention))
        for f in FEATURES:
            Image.fromarray((ev[f]['raw']*255).astype('uint8')).save(vfolder/(f+'_raw.png'))
            Image.fromarray((ev[f]['visible']*255).astype('uint8')).save(vfolder/(f+'_visible.png'))
            ev[f]['without'].save(vfolder/('without_'+f+'.png'))
        rs=regions(*base.size,scales=((3,10),));verify_crops(final,rs)
        for r in rs:
            labels,details=crop_labels(ops,ev,r['box'])
            if not accepted:labels={f:'uncertain' for f in FEATURES}
            # Unimplemented vocabulary is unknown, never encoded as absent.
            labels={f:labels.get(f,'unknown') for f in VOCABULARY}
            crop=final.crop(r['box']);crop.save(vfolder/(r['id']+'.png'))
            bricks.append(dict(id=gid+'/'+vid+'/'+r['id'],base_group=gid,partition='pilot_only',
               parent_pixel_sha256=pixel_hash(final),crop_pixel_sha256=pixel_hash(crop),box=r['box'],
               image=gid+'/'+vid+'/'+r['id']+'.png',labels=labels,evidence=details,
               text_visibility_gate=accepted,training_eligible=False))
        images.append(dict(id=gid+'/'+vid,file=gid+'/'+vid+'/image.png',enabled=enabled,
           base_group=gid,parent_pixel_sha256=pixel_hash(final),text_visibility=retention,accepted_for_label_review=accepted))
check_group_partitions(bricks)
write(out/'images.json',images);write(out/'bricks.json',bricks);write(out/'rejections.json',failure)
counts={f:{s:sum(b['labels'][f]==s for b in bricks) for s in ('present','absent','uncertain')} for f in FEATURES}
unique={}
for b in bricks:unique.setdefault(b['crop_pixel_sha256'],[]).append(b)
conflicts=[]
for h,rows in unique.items():
    for f in FEATURES:
        known={b['labels'][f] for b in rows} & {'present','absent'}
        if len(known)>1:conflicts.append(dict(hash=h,feature=f,ids=[b['id'] for b in rows]))
write(out/'duplicate_audit.json',dict(total_bricks=len(bricks),unique_pixel_crops=len(unique),
    duplicate_occurrences=len(bricks)-len(unique),contradictory_known_labels=conflicts,
    note='Counts are descriptive. Deduplicate across future partitions after base-group splitting; repeated crops are not independent support.'))
if conflicts:raise ValueError('Identical crops have contradictory known labels; label gate failed')
write(out/'status.json',dict(state='LABEL_PILOT_REVIEW_PENDING',base_groups=8,images=len(images),bricks=len(bricks),
    counts=counts,unique_pixel_crops=len(unique),rejected_images=len(failure),replays_exact=True,training_started=False,
    elapsed_seconds=time.perf_counter()-start,python_renderer_sha256=digest(project/'sourcefree_synth/renderer.py')))
# All 64 parents are visible for review, with no failed sample omitted.
for page in range(2):
    canvas=Image.new('RGB',(1100,940),'#edf1f5');d=ImageDraw.Draw(canvas)
    d.text((12,10),f'Label pilot parents {page+1}/2 | codes: straight, curve, dots | all original pixels saved',fill='black')
    for j,row in enumerate(images[page*32:(page+1)*32]):
        x=10+(j%4)*275;y=38+(j//4)*112
        d.text((x,y),row['id']+(' PASS text' if row['accepted_for_label_review'] else ' REJECT text'),fill='black')
        im=load(out/row['file']);im.thumbnail((250,84));canvas.paste(im,(x,y+19))
    canvas.save(out/f'parents_{page+1}.png')
# Deterministic first examples from each feature/state, not accuracy selection.
examples=[]
for f in FEATURES:
    for state in ('present','absent','uncertain'):
        eligible=[b for b in bricks if b['labels'][f]==state]
        if eligible:examples.append((f,state,min(eligible,key=lambda b:b['id'])))
canvas=Image.new('RGB',(1100,850),'#edf1f5');d=ImageDraw.Draw(canvas)
d.text((12,10),'Local label examples | engineering mask labels, not classifier predictions',fill='black')
for i,(f,state,row) in enumerate(examples):
    x=12+(i%3)*365;y=45+(i//3)*263
    d.text((x,y),f+' : '+state,fill='black');d.text((x,y+18),row['id'],fill='black')
    crop=load(out/row['image']);crop=crop.resize((crop.width*4,crop.height*4),Image.Resampling.NEAREST);crop.thumbnail((150,160),Image.Resampling.NEAREST)
    canvas.paste(crop,(x,y+40))
    parent=out/row['base_group']/row['id'].split('/')[1]
    mask=load(parent/(f+'_visible.png')).crop(row['box']);mask=mask.resize(crop.size,Image.Resampling.NEAREST)
    canvas.paste(mask,(x+175,y+40));d.text((x,y+210),'Image (left) | effect contribution (right)',fill='black')
    d.text((x,y+228),'visible pixels: '+str(row['evidence'][f]['visible_pixels']),fill='black')
canvas.save(out/'label_examples.png')
write(out/'example_selection.json',[dict(feature=f,state=s,id=r['id']) for f,s,r in examples])
snapshot=out/'code_snapshot'
for pth in (Path(__file__),ROOT/'feature_bank/labeled_bricks.py',ROOT/'feature_bank/regions.py'):
    target=snapshot/pth.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(pth.read_bytes())
print(counts);print('Rejected parents:',len(failure));print(out)
