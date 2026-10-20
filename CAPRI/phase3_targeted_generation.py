"""Generate a NEW targeted City corpus from the reviewed fit-only Phase3 scheme."""
from pathlib import Path
import argparse, collections, concurrent.futures, copy, csv, hashlib, importlib.util, json, os, shutil, string, subprocess, sys, time
import numpy as np
from PIL import Image
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
PKG=PROJECT/'releases/CAPRI_typography_v3'
sys.path.insert(0,str(PROJECT/'style_pipeline/phase3_reader_v2'))
from common import sha,read,write,pixels,load_image,groups,group_key,text_partition,CHARSET,CORPUS

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def draw_text(rng,n,key):
    alphabets=[string.digits,string.ascii_lowercase,string.ascii_uppercase]
    active=[alphabet for bit,alphabet in zip(key,alphabets) if bit=='1']
    assert 0<len(active)<=n
    for _ in range(10000):
        chars=[str(rng.choice(list(alphabet))) for alphabet in active]
        chars.extend(rng.choice(list(''.join(active)),size=n-len(chars)).tolist());rng.shuffle(chars);text=''.join(chars)
        # Preserve known first-reader selection/calibration/test identities.
        old=int(hashlib.sha256(('phase12-text-v1:'+text).encode()).hexdigest()[:8],16)%20
        if old<18 and text_partition(text)=='train':return text
    raise RuntimeError('Cannot sample supported training text')

def worker(a):
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    base=PROJECT/'releases/CAPRI_v2';sys.path.insert(0,str(base/'base'))
    renderer=module(PKG/'renderer.py','phase3_typography_renderer');profile=read(base/'profiles/conformer/city.json');fonts=read(PKG/'fonts.json');scheme=read(a.scheme)
    assert scheme['synthetic_reading_gate']['passed'] and scheme['source_partition']=='fit' and not scheme['validation_used_for_scheme']
    cells=scheme['joint_distribution'];weights=np.array([c['sampling_weight'] for c in cells]);assert abs(weights.sum()-1)<1e-6
    rng=np.random.default_rng(a.seed);font_rng=np.random.default_rng([a.seed,3201]);deck=[];seen=set();failures=[]
    def font():
        if not deck:deck.extend(map(int,font_rng.permutation(len(fonts))))
        return deck.pop()
    with (a.out/'labels.csv').open('w',newline='') as cf,(a.out/'manifest.jsonl').open('w') as mf:
        writer=csv.writer(cf);writer.writerow(['filename','label'])
        for i in range(a.count):
            cell=cells[int(rng.choice(len(cells),p=weights))];n=cell['length'];key=cell['groups'];mixed=bool(rng.random()<.5);gentle=bool(rng.random()<.2)
            for attempt in range(100):
                text=draw_text(rng,n,key);entry=int(rng.integers(len(profile['entries'])));style=copy.deepcopy(profile['entries'][entry]['style']);seed=int(rng.integers(2**31-1));ids=[font() for _ in text] if mixed else [font()]*n
                if mixed and len(set(ids))<2:ids[-1]=(ids[-1]+1)%len(fonts)
                paths=[str(PKG/fonts[j]['path']) for j in ids]
                controls=[dict(height_gain=float(rng.uniform(.94,1.06) if gentle else rng.uniform(.80,1.18)),width_gain=float(rng.uniform(.95,1.05) if gentle else rng.uniform(.75,1.25)),angle_delta=float(rng.uniform(-3,3) if gentle else rng.uniform(-15,15)),vertical_delta=float(rng.uniform(-.015,.015) if gentle else rng.uniform(-.06,.06)),stroke=0 if gentle else int(rng.random()<.2),effect='filled' if gentle else str(rng.choice(['filled','bold','hollow'],p=[.75,.15,.10]))) for _ in text]
                style['_typography']=controls;style['spacing_gain']=float(style.get('spacing_gain',1)*rng.uniform(.9,1.1))
                try:
                    im,mask,recipe=renderer.render(style,text,paths,seed,profile['variant']);h=pixels(im)
                    assert ''.join(g['character'] for g in recipe['glyphs'])==text and min(g['visible_fraction'] for g in recipe['glyphs'])>=.55
                    if h in seen:raise ValueError('duplicate pixels')
                    if i<20:assert renderer.render(style,text,paths,seed,profile['variant'])[0].tobytes()==im.tobytes()
                except ValueError as e:failures.append(dict(index=i,attempt=attempt,error=str(e)));continue
                break
            else:raise RuntimeError('100 failed rendering attempts')
            seen.add(h);name=f'images/{i:07d}.png';im.save(a.out/name);writer.writerow([name,text]);mf.write(json.dumps(dict(filename=name,label=text,length=n,groups=key,partition='train',font_ids=ids,font_families=[fonts[j]['family'] for j in ids],mixed=mixed,gentle=gentle,seed=seed,profile_entry=entry,spacing_gain=style['spacing_gain'],pixel_sha256=h,recipe=recipe))+'\n')
            if (i+1)%500==0:mf.flush();cf.flush();write(a.out/'status.json',dict(state='generating',generated=i+1,pid=os.getpid()))
    write(a.out/'render_failures.json',failures);write(a.out/'status.json',dict(state='complete',generated=a.count,pid=os.getpid()))

def coordinate(a):
    assert not a.out.exists();scheme=read(a.scheme);assert scheme['synthetic_reading_gate']['passed'];assert scheme['source_partition']=='fit' and scheme['images']==1599 and not scheme['validation_used_for_scheme']
    assert sha(PKG/'INTEGRITY.json')==read(CORPUS/'protocol.json')['package_manifest_sha256']
    for rel,digest in read(PKG/'INTEGRITY.json').items():assert sha(PKG/rel)==digest
    for rel,digest in read(PROJECT/'releases/CAPRI_v2/INTEGRITY.json').items():assert sha(PROJECT/'releases/CAPRI_v2'/rel)==digest
    fonts=read(PKG/'fonts.json')
    for f in fonts:assert sha(PKG/f['path'])==f['sha256'] and sha(PKG/f['license_path'])==f['license_sha256']
    a.out.mkdir();(a.out/'shards').mkdir();(a.out/'logs').mkdir();shutil.copy2(a.scheme,a.out/'inferred_scheme.json');shutil.copy2(__file__,a.out/'generator_snapshot.py')
    root=a.scheme.parent
    for name in ['comparison.json','upstream_provenance.json']:shutil.copy2(root/name,a.out/name)
    if (root/'scheme_review.json').exists():shutil.copy2(root/'scheme_review.json',a.out/'scheme_review.json')
    shutil.copy2(root/'data/audit.json',a.out/'source_data_audit.json');shutil.copy2(root/'data/partitions.jsonl',a.out/'reader_partitions.jsonl')
    settings=dict(stage='Official CAPRI Phase3 targeted generation',count=a.count,scheme_sha256=sha(a.scheme),scheme_fit_only=True,model_checkpoint_hashes=scheme['models'],profile_sha256=sha(PROJECT/'releases/CAPRI_v2/profiles/conformer/city.json'),font_manifest_sha256=sha(PKG/'fonts.json'),renderer_sha256=sha(PKG/'renderer.py'),fonts=35,mixed_fraction=.5,typography='same broad Phase2 ranges',refinement='off',seed=a.seed,real_labels_used=False,generation_source_sha256=sha(__file__),uncertainty_prior_mass=scheme['uncertainty_prior_mass'],lengths=list(range(2,9)),group_order=['digits','lowercase','uppercase'],alphabet=CHARSET,shortcut_limit=scheme['shortcut_limit'],partition_rule='phase12 train AND phase3-reader train text hashes',not_a_real_accuracy_claim=True)
    write(a.out/'protocol.json',settings)
    def status(state,**extra):
        row=dict(state=state,pid=os.getpid(),time=time.time(),target=a.count,dataset=str(a.out),**extra);write(a.out/'status.json',row);write(PROJECT/'monitor/capri_phase3_city_status.json',row)
    count_shards=10;shards=[(i,a.count//count_shards+(i<a.count%count_shards)) for i in range(count_shards)]
    def run_shard(item):
        i,n=item
        with (a.out/'logs'/f'{i:02d}.log').open('w') as log:
            subprocess.run([sys.executable,__file__,'--worker','--scheme',str(a.out/'inferred_scheme.json'),'--out',str(a.out/'shards'/f'{i:02d}'),'--count',str(n),'--seed',str(a.seed+i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(run_shard,item) for item in shards}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:future.result()
            progress=sum(read(a.out/'shards'/f'{i:02d}'/'status.json').get('generated',0) for i,n in shards if (a.out/'shards'/f'{i:02d}'/'status.json').exists());status('generating_targeted_dataset',generated=progress)
    seen=set();cells=collections.Counter();chars=collections.Counter();fc=collections.Counter();modes=collections.Counter();minimum_visibility=1.;total=0
    old_hashes=set()
    for line in (root/'data/partitions.jsonl').open():old_hashes.add(json.loads(line)['pixel_sha256'])
    real_hashes={r['pixel_sha256'] for values in read(root/'data/real_locators.json').values() for r in values}
    with (a.out/'labels.csv').open('w',newline='') as cf,(a.out/'manifest.jsonl').open('w') as mf:
        writer=csv.writer(cf);writer.writerow(['filename','label'])
        for i,n in shards:
            shard=a.out/'shards'/f'{i:02d}';labels=list(csv.DictReader((shard/'labels.csv').open()));num=0
            for line in (shard/'manifest.jsonl').open():
                r=json.loads(line);assert labels[num]==dict(filename=r['filename'],label=r['label']);path=shard/r['filename'];im=load_image(path);h=pixels(im)
                assert h==r['pixel_sha256'] and h not in seen and h not in old_hashes and h not in real_hashes;seen.add(h)
                t=r['label'];assert len(t)==r['length'] and group_key(t)==r['groups'] and set(t)<=set(CHARSET);assert text_partition(t)=='train'
                assert ''.join(g['character'] for g in r['recipe']['glyphs'])==t;assert len(r['font_ids'])==len(t)
                for g in r['recipe']['glyphs']:
                    assert g['visible_fraction']>=.55;minimum_visibility=min(minimum_visibility,g['visible_fraction']);x0,y0,x1,y1=g['box'];assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height
                assert (len(set(r['font_ids']))>1)==r['mixed'];r['filename']=str(path.relative_to(a.out)).replace('\\','/');writer.writerow([r['filename'],t]);mf.write(json.dumps(r)+'\n');cells[str(len(t))+':'+r['groups']]+=1;chars.update(t);fc.update(r['font_families']);modes['mixed' if r['mixed'] else 'single']+=1;total+=1;num+=1
                if total%5000==0:status('auditing_targeted_dataset',audited=total)
            assert num==n
    assert total==a.count and len(fc)==35 and set(chars)==set(CHARSET)
    expected={str(c['length'])+':'+c['groups']:c['sampling_weight'] for c in scheme['joint_distribution']};assert set(cells)==set(expected)
    sampling={k:dict(count=cells[k],observed_fraction=cells[k]/total,target_fraction=v) for k,v in expected.items()}
    audit=dict(passed=True,images=total,unique_pixels=len(seen),all_images_rehashed=True,label_metadata_consistency=True,minimum_glyph_visibility=minimum_visibility,zero_overlap_with_initial_or_real_pixels=True,fonts=len(fc),font_glyph_counts=dict(fc),modes=dict(modes),character_counts=dict(chars),joint_sampling=sampling,scheme_sha256=sha(a.out/'inferred_scheme.json'),labels_sha256=sha(a.out/'labels.csv'),manifest_sha256=sha(a.out/'manifest.jsonl'),real_accuracy_verified=False)
    write(a.out/'audit.json',audit);write(a.out/'sampling_statistics.json',sampling);status('complete',generated=total,audited=total,audit_passed=True)
    (a.out/'README.md').write_text('# CAPRI official Phase3 targeted City corpus\n\nA separate labelled synthetic dataset from frozen fit-only reader predictions. All35 audited fonts and broad Phase2 typography retained; optional refiners off. Read inferred_scheme.json, protocol.json, audit.json and sampling_statistics.json. Inferred predictions are not real ground truth; uncertainty remains in the sampling distribution. Count-dependent renderer layout remains a limitation. No production recognizer trained. The three first-reader heldout fonts are included here, so that original font test is no longer a heldout-font test for a future recognizer trained on this corpus. Text boundaries are preserved; future recognizer evaluation needs its own predeclared protocol and unexposed real data for generalization claims.\n')
    print('COMPLETE',a.out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scheme',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=100000);p.add_argument('--seed',type=int,default=202609290100);p.add_argument('--worker',action='store_true');a=p.parse_args()
    try:worker(a) if a.worker else coordinate(a)
    except BaseException as e:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(e),time=time.time()));write(a.out/'status.json',dict(state='failed',error=repr(e),time=time.time()))
        raise
