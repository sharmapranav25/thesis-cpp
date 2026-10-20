import common as U
import numpy as np
from PIL import Image,ImageDraw
import render_control as R
from comparison import Comparator
def main():
    out=U.ROOT/'inspection';out.mkdir(exist_ok=True)
    split=U.read(U.WORK/'feature_ablation_20261004/run_03/City/target_splits.json')
    images=[np.asarray(Image.open(r['source']).convert('RGB')) for r in split['fit'][:12]]
    model=Comparator();f=model.extract(images)
    descriptions=model.describe(images[:2]);U.dump(out/'descriptions.json',descriptions)
    recipes=U.read(U.WORK/'feature_ablation_20261004/run_03/City/fit_recipes.json')[:12]
    original=[R.original(r) for r in recipes]
    R.install();rendered=[R.render((r,R.NEUTRAL)) for r in recipes]
    assert all(np.array_equal(a['image'],b['image']) for a,b in zip(original,rendered))
    U.dump(out/'neutral_check.json',dict(exact_replays=len(recipes)))
    sf=model.extract([r['image'] for r in rendered])
    sheet=Image.new('RGB',(256*4,90*12),'white');d=ImageDraw.Draw(sheet)
    for i,(im,m,row,sm) in enumerate(zip(images,f['mask'],rendered,sf['mask'])):
        a=np.asarray(Image.fromarray(im).resize((256,64)))
        b=np.asarray(Image.fromarray(row['image']).resize((256,64)))
        for j,(pixels,mask) in enumerate([(a,None),(a,m),(b,None),(b,sm)]):
            if mask is not None:pixels=np.clip(pixels*.55+np.stack([mask*255,np.zeros_like(mask),np.zeros_like(mask)],2)*.45,0,255).astype(np.uint8)
            sheet.paste(Image.fromarray(pixels),(256*j,i*90+20))
        d.text((5,i*90+3),f'{i}: target | inferred target text | hybrid render | inferred rendered text',fill='black')
    sheet.save(out/'masks.jpg')
    U.dump(out/'mask_diagnostic.json',dict(real_mean=f['mask'].mean(),synthetic_mean=sf['mask'].mean(),synthetic_iou=[float(((m>.5)&(np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>0)).sum()/max(1,((m>.5)|(np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>0)).sum())) for m,r in zip(sf['mask'],rendered)]))
    print('Inspection complete, neutral pixels match exactly.',flush=True)
if __name__=='__main__':main()
