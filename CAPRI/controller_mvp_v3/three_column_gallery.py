"""Presentation only: original pixels, neutral hybrid recipe, saved full-v3 output."""
from boot import ROOT,U
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import controls as C
from run import save_rows

def main():
    dest=ROOT/'three_column_gallery';dest.mkdir(exist_ok=True);C.install();manifest={}
    fontpath='C:/Windows/Fonts/arial.ttf'
    titlefont=ImageFont.truetype(fontpath,28);headerfont=ImageFont.truetype(fontpath,20);smallfont=ImageFont.truetype(fontpath,15)
    for family in ['City','CryptBB','DarkBay','Outlined','SimpleCaptcha']:
        run=ROOT/(family+'_run_01');out=dest/family;out.mkdir(exist_ok=True)
        split=U.read(run/'target_splits.json');group='fresh' if split['fresh'] else 'test'
        targets=split[group][:8];recipes=U.read(run/'test_recipes.json')[:8];controller=U.read(run/'full/labels.json')[:8]
        digest=U.sha(run/'frozen_policies.json')
        hybrid=[C.render((recipe,C.expand(C.R.NEUTRAL))) for recipe in recipes]
        save_rows(out,'hybrid_start',hybrid)
        sheet=Image.new('RGB',(900,920),'#f5f6f8');draw=ImageDraw.Draw(sheet)
        name='Simple Captcha (Kaggle)' if family=='SimpleCaptcha' else family
        draw.text((12,10),name,font=titlefont,fill='#172033')
        for j,label in enumerate(['Original','Hybrid guess','Controller v3']):draw.text((j*300+12,53),label,font=headerfont,fill='#172033')
        rows=[]
        for i,(target,h,c,recipe) in enumerate(zip(targets,hybrid,controller,recipes)):
            actual=np.asarray(Image.open(target['source']).convert('RGB'))
            refined=np.asarray(Image.open(run/c['image']).convert('RGB'))
            assert hashlib.sha256(actual.tobytes()).hexdigest()==target['pixel_sha256']
            assert hashlib.sha256(refined.tobytes()).hexdigest()==c['pixel_sha256']
            assert h['seed']==c['seed']==recipe['seed']
            for j,pixels in enumerate([actual,h['image'],refined]):
                im=ImageOps.contain(Image.fromarray(pixels),(288,86),Image.Resampling.NEAREST)
                sheet.paste(im,(j*300+(300-im.width)//2,84+i*100+(86-im.height)//2))
            rows.append(dict(target=target,hybrid_pixel_sha256=h['pixel_sha256'],controller_pixel_sha256=c['pixel_sha256'],seed=h['seed']))
        draw.text((12,889),'Scheme-level samples: generated text differs from the originals. First 8 examples.',font=smallfont,fill='#465364')
        sheet.save(dest/(family+'.png'))
        assert U.sha(run/'frozen_policies.json')==digest
        manifest[family]=dict(target_split=group,rows=rows,frozen_policy_sha256=digest,controller='Previously saved full v3 outputs, unmodified',hybrid='Original test recipes rendered with neutral controls; paired seeds with controller')
        print(family,'saved',flush=True)
    U.dump(dest/'manifest.json',manifest)

if __name__=='__main__':main()
