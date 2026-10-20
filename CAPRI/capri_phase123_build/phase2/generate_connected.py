"""Phase2 broad typography over frozen CAPRI Phase1 profiles; no model training."""
from pathlib import Path
import argparse,copy,csv,hashlib,importlib.util,json,sys,collections
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT.parent/'phase1');source=p.add_mutually_exclusive_group(required=True);source.add_argument('--family');source.add_argument('--profile',type=Path);p.add_argument('--count',type=int,default=210);p.add_argument('--seed',type=int,required=True);p.add_argument('--partition',choices=['train','validation','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--lengths',nargs='+',type=int,default=list(range(2,9)));p.add_argument('--mixed-fraction',type=float,default=.5)
 a=p.parse_args()
 assert a.count>0 and 0<=a.mixed_fraction<=1 and all(2<=n<=8 for n in a.lengths)
 # Verify every byte recorded by the frozen upstream release before use.
 for rel,digest in json.loads((a.base/'INTEGRITY.json').read_text()).items():assert sha(a.base/rel)==digest,rel
 sys.path.insert(0,str(a.base/'base'));basegen=module(a.base/'generation.py','basegen');original=module(a.base/'renderer.py','original_renderer');renderer=module(ROOT/'renderer.py','typography_renderer')
 fonts=json.loads((ROOT/'fonts.json').read_text())
 for f in fonts:
  assert sha(ROOT/f['path'])==f['sha256'];assert sha(ROOT/f['license_path'])==f['license_sha256']
 profile_path=a.profile if a.profile else a.base/'profiles/conformer'/(a.family+'.json');profile=json.loads(profile_path.read_text());rng=np.random.default_rng(a.seed);font_rng=np.random.default_rng(np.random.SeedSequence([a.seed,3201]));deck=[]
 def drawfont():
  if not deck:deck.extend(map(int,font_rng.permutation(len(fonts))))
  return deck.pop()
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
 cells=[(b,n) for b in ('numeric','alphabetic','alphanumeric') for n in sorted(set(a.lengths))];seen=set();counts=collections.Counter();font_counts=collections.Counter();image_font_counts=collections.Counter();modes=collections.Counter();fails=[];previews=[];background_checks=0
 protocol=dict(method='CAPRI Phase2 broad typography v3',base=str(a.base),base_manifest_sha256=sha(a.base/'INTEGRITY.json'),profile_sha256=sha(profile_path),font_manifest_sha256=sha(ROOT/'fonts.json'),seed=a.seed,partition=a.partition,fonts=len(fonts),font_sampling='shuffled uniform decks; accepted exposure audited after retries',mixed_fraction=a.mixed_fraction,lengths=sorted(set(a.lengths)),length_inference=False,length_source='explicit configuration, not real estimation',refinement='off for new typography ablation; old adapters not refitted',real_labels_used=False,training=False,background_policy='frozen profile operations; exact operation equality checked against original renderer',gentle_fraction=.2)
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 with (a.out/'labels.csv').open('w',newline='') as cf,(a.out/'manifest.jsonl').open('w') as mf:
  writer=csv.writer(cf);writer.writerow(['filename','label'])
  for i in range(a.count):
   bucket,n=cells[i%len(cells)];mixed=bool(rng.random()<a.mixed_fraction);gentle=bool(rng.random()<.2)
   for attempt in range(100):
    text,sub=basegen.make_text(rng,bucket,n,a.partition);entry=int(rng.integers(len(profile['entries'])));style=copy.deepcopy(profile['entries'][entry]['style']);seed=int(rng.integers(2**31-1));ids=[drawfont() for _ in text] if mixed else [drawfont()]*n
    if mixed and len(set(ids))<2:ids[-1]=(ids[-1]+1)%len(fonts)
    paths=[str(ROOT/fonts[j]['path']) for j in ids]
    controls=[dict(height_gain=float(rng.uniform(.94,1.06) if gentle else rng.uniform(.80,1.18)),width_gain=float(rng.uniform(.95,1.05) if gentle else rng.uniform(.75,1.25)),angle_delta=float(rng.uniform(-3,3) if gentle else rng.uniform(-15,15)),vertical_delta=float(rng.uniform(-.015,.015) if gentle else rng.uniform(-.06,.06)),stroke=0 if gentle else int(rng.random()<.2),effect='filled' if gentle else str(rng.choice(['filled','bold','hollow'],p=[.75,.15,.10]))) for _ in text]
    style['_typography']=controls;style['spacing_gain']=float(style.get('spacing_gain',1)*rng.uniform(.9,1.1))
    try:
     image,mask,recipe=renderer.render(style,text,paths,seed,profile['variant'])
     digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
     if digest in seen:raise ValueError('duplicate pixels')
     assert len(recipe['glyphs'])==n and ''.join(g['character'] for g in recipe['glyphs'])==text
     # A text-free background is not emitted by old code; its exact drawing
     # operations are compared with a render using the same adjusted layout.
     if i<42:
      clean=copy.deepcopy(style);clean.pop('_typography');_,_,baseline=original.render(clean,text,paths[0],seed,profile['variant'])
      assert recipe['operations']==baseline['operations'],'Background operation regression'
      replay=renderer.render(style,text,paths,seed,profile['variant'])[0]
      assert replay.tobytes()==image.tobytes(),'Replay regression'
      background_checks+=1
    except ValueError as e:fails.append(dict(index=i,attempt=attempt,error=str(e)));continue
    break
   else:raise RuntimeError('No valid labelled render after 100 attempts')
   seen.add(digest);name=f'images/{i:07d}.png';image.save(a.out/name);writer.writerow([name,text]);font_counts.update(fonts[j]['family'] for j in ids);image_font_counts.update(set(fonts[j]['family'] for j in ids));modes['mixed' if mixed else 'single']+=1;counts[bucket+'_'+str(n)]+=1
   row=dict(filename=name,label=text,length=n,bucket=bucket,partition=a.partition,font_ids=ids,font_families=[fonts[j]['family'] for j in ids],mixed=mixed,gentle=gentle,seed=seed,profile_entry=entry,pixel_sha256=digest,recipe=recipe);mf.write(json.dumps(row)+'\n')
   if i<42:previews.append((image.copy(),text+' | '+('mixed' if mixed else 'single')))
 gallery=Image.new('RGB',(1200,120*((len(previews)+3)//4)),'white');d=ImageDraw.Draw(gallery)
 for i,(im,label) in enumerate(previews):
  im.thumbnail((285,90));x=300*(i%4);y=120*(i//4);d.text((x+5,y+3),label,fill='black');gallery.paste(im,(x+5,y+25))
 gallery.save(a.out/'preview.png')
 audit=dict(state='complete',images=a.count,unique_pixels=len(seen),fonts=len(fonts),font_glyph_counts=dict(font_counts),font_image_counts=dict(image_font_counts),modes=dict(modes),cells=dict(counts),retries=len(fails),background_and_replay_checks=background_checks,minimum_glyph_visibility=.55,recognition_benefit_verified=False)
 (a.out/'audit.json').write_text(json.dumps(audit,indent=2));(a.out/'failures.json').write_text(json.dumps(fails,indent=2));print(json.dumps(audit))
if __name__=='__main__':main()
