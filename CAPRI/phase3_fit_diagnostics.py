"""Frozen fit-only sensitivity and native-size stratification, with no real labels."""
from pathlib import Path
import argparse,collections,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(PROJECT/'style_pipeline/phase3_reader_v2'))
from common import read,write,sha,load_image
from models import Reader
from train import batch

def main(a):
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    loc=read(a.root/'data/real_locators.json')['fit'];assert len(loc)==1599
    original=read(PROJECT/'runs/pixel_statistics_01/city/split.json')['fit'];assert [r['pixel_sha256'] for r in original]==[r['pixel_sha256'] for r in loc]
    cache=np.load(a.root/'data/fit.npy',mmap_mode='r');ids=np.array(sorted(range(1599),key=lambda i:loc[i]['pixel_sha256'])[:192])
    scheme=read(a.root/'inferred_scheme.json');all_results={};display={}
    for name,digest in scheme['models'].items():
        run=a.root/name;assert sha(run/'best.pt')==digest;checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=checkpoint['protocol'];m=Reader(cfg['model'],cfg['auxiliary'],cfg['seed']).to('cuda').eval();m.load_state_dict(checkpoint['state_dict']);m.requires_grad_(False)
        rows=[json.loads(l) for l in (run/'fit_predictions.jsonl').open()];assert [r['pixel_sha256'] for r in rows]==[r['pixel_sha256'] for r in loc]
        strata=collections.defaultdict(collections.Counter)
        for r,o in zip(rows,original):strata[f"{o['width']}x{o['height']}"][r['decoded_length']]+=1
        views={}
        for gain in [.75,1.]:
            predictions=[]
            for start in range(0,len(ids),64):
                x=batch(cache,ids[start:start+64],'cuda')
                if gain!=1.:
                    width=round(192*gain);x=F.interpolate(x,size=(32,width),mode='bilinear',align_corners=False);x=F.pad(x,((192-width)//2,192-width-(192-width)//2),value=1.)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p,_,_=m.predict(x)
                predictions.extend(p)
            views[str(gain)]=predictions
        before=views['1.0'];after=views['0.75'];result=dict(checkpoint_sha256=digest,fit_only=True,sample_selection='first192 SHA256-sorted fit image identities',sample_ids=[loc[i]['id'] for i in ids],predicted_length_histogram_by_native_canvas={k:dict(v) for k,v in strata.items()},width_compression_probe=dict(scale=.75,string_change_fraction=float(np.mean([a!=b for a,b in zip(before,after)])),length_change_fraction=float(np.mean([len(a)!=len(b) for a,b in zip(before,after)])),before_length_histogram=dict(collections.Counter(map(len,before))),after_length_histogram=dict(collections.Counter(map(len,after)))),interpretation='Unlabelled sensitivity test; compression alters effective glyph resolution/style. Neither view is declared correct, no real calibration or prediction replacement.',real_accuracy=None)
        all_results[name]=result;display[name]=before
        with (a.out/(name+'_probe.jsonl')).open('w') as f:
            for i,b,c in zip(ids,before,after):f.write(json.dumps(dict(id=loc[i]['id'],baseline=b,compressed=c))+'\n')
        assert sha(run/'best.pt')==digest;del m;torch.cuda.empty_cache()
    write(a.out/'diagnostics.json',all_results)
    # Reviewable fit-only gallery: every displayed string is explicitly a model prediction.
    board=Image.new('RGB',(1200,8*145),'white');draw=ImageDraw.Draw(board)
    for j,i in enumerate(ids[:24]):
        x=(j%3)*400;y=(j//3)*145;im=load_image(loc[i]['path']);im.thumbnail((385,90));board.paste(im,(x+5,y+4));draw.text((x+5,y+92),loc[i]['id']+' | predicted:',fill='black')
        draw.text((x+5,y+107),' / '.join(name.split('_')[0]+':'+display[name][j] for name in display),fill='blue')
    board.save(a.out/'fit_predictions_gallery.png');print(json.dumps(all_results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
