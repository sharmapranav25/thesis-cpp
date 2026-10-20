"""Rehash all input pixels, establish text boundaries, audit layout shortcuts."""
import argparse, collections, csv, importlib.metadata, sys
from pathlib import Path
import numpy as np
from common import *

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(state='auditing',pid=os.getpid()))
    assert read(CORPUS/'status.json')['state']=='complete' and read(CORPUS/'audit.json')['passed']
    assert not (CORPUS/'failure.json').exists()
    fonts=read(PROJECT/'releases/CAPRI_typography_v3/fonts.json')
    # Predeclared hash-order held-out font families, chosen without any model score.
    held=sorted(range(len(fonts)),key=lambda j:hashlib.sha256(fonts[j]['family'].encode()).hexdigest())[:3]
    csvrows=list(csv.DictReader((CORPUS/'labels.csv').open(newline='')))
    cache=np.lib.format.open_memmap(out/'synthetic.npy',mode='w+',dtype=np.uint8,shape=(len(csvrows),32,192,3))
    records=[]; seen=set(); cases=collections.Counter(); parts=collections.Counter(); features=[]; lengths=[]; perlength=collections.defaultdict(list)
    with (CORPUS/'manifest.jsonl').open() as f:
        for i,line in enumerate(f):
            r=json.loads(line); t=r['label']; assert csvrows[i]==dict(filename=r['filename'],label=t)
            assert set(t)<=set(CHARSET) and 2<=len(t)<=8 and r['length']==len(t)
            im=load_image(CORPUS/r['filename']); h=pixels(im); assert h==r['pixel_sha256'] and h not in seen;seen.add(h)
            glyphs=r['recipe']['glyphs'];assert ''.join(g['character'] for g in glyphs)==t and min(g['visible_fraction'] for g in glyphs)>=.55
            cache[i]=np.asarray(im.resize(SIZE,Image.Resampling.BICUBIC))
            partition=text_partition(t); withheld=bool(set(r['font_ids']) & set(held))
            # A font-heldout test also has heldout text. Other withheld-font records are excluded.
            role=('font_test' if partition=='test' else 'excluded_font') if withheld else partition
            cases[group_key(t)]+=1;parts[role]+=1
            boxes=np.asarray([g['box'] for g in glyphs]); width=boxes[:,2]-boxes[:,0];height=boxes[:,3]-boxes[:,1]
            centers=(boxes[:,2]+boxes[:,0])/2; gaps=np.diff(centers)
            feat=[im.width,im.height,float(width.mean()),float(height.mean()),float(gaps.mean()),float(np.ptp(centers))]
            features.append(feat); lengths.append(len(t));perlength[len(t)].append(feat)
            records.append(dict(index=i,path=str(CORPUS/r['filename']),label=t,pixel_sha256=h,partition=role,text_partition=partition,group=group_key(t),font_ids=r['font_ids']))
            if (i+1)%5000==0:write(out/'status.json',dict(state='auditing',pid=os.getpid(),images=i+1));print('audited',i+1,flush=True)
    assert len(records)==100000
    cache.flush(); del cache
    with (out/'partitions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    real=read(PROJECT/'runs/pixel_statistics_01/city/split.json'); real_records={}; hashes={}
    assert len(real['fit'])==1599 and len(real['validation'])==400
    for part in ['fit','validation']:
        rows=[]; hashes[part]=set(); arr=np.lib.format.open_memmap(out/(part+'.npy'),mode='w+',dtype=np.uint8,shape=(len(real[part]),32,192,3))
        for i,r in enumerate(real[part]):
            path=Path(r['path']);assert sha(path)==r['file_sha256'];im=load_image(path);h=pixels(im);assert h==r['pixel_sha256'];assert im.size==(r['width'],r['height']);assert h not in hashes[part];hashes[part].add(h)
            arr[i]=np.asarray(im.resize(SIZE,Image.Resampling.BICUBIC));rows.append(dict(id=f'{part}_{i:04d}',path=str(path),pixel_sha256=h,file_sha256=r['file_sha256']))
        arr.flush();del arr;real_records[part]=rows
    assert not hashes['fit'] & hashes['validation'];assert not seen & (hashes['fit']|hashes['validation'])
    write(out/'real_locators.json',real_records)
    # No real transcription is derived from paths, including answer-looking names.
    from sklearn.ensemble import ExtraTreesClassifier
    x=np.asarray(features);y=np.asarray(lengths);train=np.array([r['partition']=='train' for r in records]);test=np.array([r['partition']=='test' for r in records])
    model=ExtraTreesClassifier(n_estimators=80,max_depth=12,n_jobs=4,random_state=731).fit(x[train],y[train])
    shortcut=dict(features=['canvas_width','canvas_height','mean_glyph_width','mean_glyph_height','mean_center_gap','occupied_center_span'],test_accuracy=float((model.predict(x[test])==y[test]).mean()),majority_baseline=float(max(collections.Counter(y[test]).values())/test.sum()),feature_importances=model.feature_importances_.tolist(),mean_by_length={str(k):np.mean(v,axis=0).tolist() for k,v in perlength.items()},claim='Metadata diagnostic only. Renderer retains count-dependent spacing and width; NOT shortcut-resistant counting.')
    write(out/'shortcut_audit.json',shortcut)
    write(out/'audit.json',dict(passed=True,images=len(records),unique_pixels=len(seen),case_group_counts=dict(cases),partitions=dict(parts),heldout_font_ids=held,heldout_font_families=[fonts[j]['family'] for j in held],real_counts={k:len(v) for k,v in real_records.items()},real_labels_available=False,real_accuracy_permitted=False,manifest_sha256=sha(CORPUS/'manifest.jsonl'),labels_sha256=sha(CORPUS/'labels.csv'),protocol_sha256=sha(CORPUS/'protocol.json'),split_sha256=sha(PROJECT/'runs/pixel_statistics_01/city/split.json'),partitions_sha256=sha(out/'partitions.jsonl'),text_partition_rule='SHA256 capri-phase3-city-v1, 80/8/6/6; exclude any heldout font from fitting/selection/calibration',shortcut=shortcut))
    write(out/'status.json',dict(state='complete',pid=os.getpid(),images=len(records)))
    print(json.dumps(read(out/'audit.json'),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a.out)
    except BaseException as e:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(e)));write(a.out/'status.json',dict(state='failed',error=repr(e)))
        raise
