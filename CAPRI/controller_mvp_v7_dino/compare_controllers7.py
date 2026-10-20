"""Frozen custom-controller and DINO profiles on matched synthetic recipes/seeds."""
from setup7 import ROOT,V4,U,FAMILIES
import render7 as R
from run import save_rows
from PIL import Image,ImageDraw,ImageFont,ImageOps

def main():
    out=ROOT/'controller_comparison';out.mkdir(exist_ok=True);R.install();manifest={}
    font='C:/Windows/Fonts/arial.ttf';title=ImageFont.truetype(font,27);header=ImageFont.truetype(font,20);small=ImageFont.truetype(font,15)
    for family in FAMILIES:
        dest=ROOT/family;local=out/family;local.mkdir(exist_ok=True)
        chosen=U.read(dest/'selected.json');assert chosen['bank']=='hybrid'
        recipes=U.read(dest/'recipe_banks.json')['hybrid']['test'][:6]
        profile=V4/'collections'/family/'single.json';controls=U.read(profile)
        rows=[R.render((r,controls)) for r in recipes];save_rows(local,'custom',rows)
        refs=U.read(dest/'target_splits.json')['test'];dino=U.read(dest/'dino_v2/labels.json')[:6]
        assert all(a['seed']==b['seed'] for a,b in zip(rows,dino))
        sheet=Image.new('RGB',(900,704),'#f4f6f8');draw=ImageDraw.Draw(sheet)
        draw.text((12,10),family+' | controller comparison',font=title,fill='#172033')
        for j,name in enumerate(['Original target','Our controller','DINO controller v2']):
            draw.text((12+300*j,50),name,font=header,fill='#172033')
        for i in range(6):
            images=[Image.open(refs[i]['source']).convert('RGB'),Image.fromarray(rows[i]['image']),Image.open(dest/dino[i]['image']).convert('RGB')]
            for j,im in enumerate(images):
                tile=ImageOps.contain(im,(286,80),Image.Resampling.NEAREST)
                sheet.paste(tile,(300*j+(300-tile.width)//2,82+98*i+(80-tile.height)//2))
        draw.text((12,678),'First 6 samples; matched starting recipes/seeds. Generated strings may differ.',font=small,fill='#465364')
        sheet.save(out/f'{family}.png')
        manifest[family]=dict(custom_profile=str(profile),custom_profile_sha256=U.sha(profile),dino_selected_sha256=U.sha(dest/'selected.json'),
            samples=6,matched_recipe_bank=True,matched_seeds=True,custom_complete=sum(r['complete'] for r in rows),custom_usable=sum(r['usable'] for r in rows))
    U.dump(out/'manifest.json',dict(note='Our controller = frozen custom Controller 1 (MVP4); DINO = frozen DINO v2 (MVP7). Existing profiles only, no search or selection changes. Training/search budgets differ.',families=manifest))
    print('Created all five matched comparisons.')

if __name__=='__main__':main()
