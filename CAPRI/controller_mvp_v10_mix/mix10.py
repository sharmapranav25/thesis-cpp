"""Controller 3: frozen custom + frozen DINO v2, calibrated 50/50 scoring."""
from pathlib import Path
import sys,copy,json,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controller_mvp_v7_dino'))
from setup7 import U,WORK,FAMILIES,V3,V4,WEIGHTS_PATH
from perception3 import Perception,Objective as CustomObjective
from perception7 import Comparator,Objective7
from build7 import fresh
from run import load_images,save_rows
import render7 as R
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from concurrent.futures import ProcessPoolExecutor

TEXT=['size','width','spacing','rotation','jitter','weight','font','text_palette','color_mode','length','jitter_absolute','rotation_absolute','bold_extra','font_pool',*R.DEFAULT]
# Background is left with its own parent; effect colors follow the background parent.
FIELDS=['size_absolute','width_absolute','spacing_absolute','rotation_absolute','jitter_absolute','length','font','text_palette','background','color_mode','glyph_outline','frame_width']

def key(p):return json.dumps(p,sort_keys=True)

def crossover(text,bg):
    p=copy.deepcopy(bg)
    for k in TEXT:p[k]=copy.deepcopy(text[k])
    glyph=R.C.R.R.GLYPH_EFFECTS
    for name in R.C.R.R.SAMPLED:
        source=text if name in glyph else bg
        if name in source['effects']:p['effects'][name]=source['effects'][name]
        else:p['effects'].pop(name,None)
    if p['keep_only'] is not None or text['keep_only'] is not None:
        p['keep_only']=[name for name in R.C.R.R.SAMPLED if (text if name in glyph else bg)['keep_only'] is None or name in (text if name in glyph else bg)['keep_only']]
    p['effect_styles']={**{k:v for k,v in bg['effect_styles'].items() if k not in glyph},**{k:v for k,v in text['effect_styles'].items() if k in glyph}}
    return p

def candidates(custom,dino):
    rng=np.random.default_rng(101019);pool=[('neutral',R.expand({})),('custom',custom),('dino',dino),('custom_text_dino_bg',crossover(custom,dino)),('dino_text_custom_bg',crossover(dino,custom))]
    for i in range(24):
        p=copy.deepcopy(custom if i%2 else dino);other=dino if i%2 else custom
        for k in p:
            if k not in ['effects','effect_styles'] and rng.random()<.5:p[k]=copy.deepcopy(other[k])
        for effect in R.C.R.R.SAMPLED:
            if rng.random()<.5:
                if effect in other['effects']:p['effects'][effect]=other['effects'][effect]
                else:p['effects'].pop(effect,None)
        pool.append(('crossover',p))
    return pool

class Mixed:
    def __init__(self):self.custom=Perception();self.dino=Comparator()
    def extract(self,images):return self.custom.extract_pair(images)[1],self.dino.extract(images)

class Objective:
    def __init__(self,target,baseline):
        self.custom=CustomObjective(target[0],baseline[0]);self.dino=Objective7(target[1],baseline[1])
        dummy=[dict(complete=True,usable=True,strict=True)]*len(baseline[0]['whole'])
        self.calibration=[max(.05,self.custom.score(baseline[0],target[0],dummy)['appearance']),max(.05,self.dino.score(baseline[1],target[1],dummy)['appearance'])]
    def score(self,f,target,rows):
        a=self.custom.score(f[0],target[0],rows);b=self.dino.score(f[1],target[1],rows)
        ca=a['appearance']/self.calibration[0];da=b['appearance']/self.calibration[1]
        valid=4*(1-a['complete'])+2*(1-a['usable']);guard=b['contrast_penalty']
        return dict(custom_appearance=ca,dino_appearance=da,appearance=.5*(ca+da),objective=.5*(ca+da)+guard+valid,
            custom_objective=ca+valid,dino_objective=da+guard+valid,contrast_penalty=guard,complete=a['complete'],usable=a['usable'],strict=a['strict'],
            audit_pixel_distance=a['audit'],custom_detail=a,dino_detail=b)

def render(pool,recipes,p):
    rows=list(pool.map(R.safe_render,[(r,p) for r in recipes]))
    for row,r in zip(rows,recipes):
        if 'image' not in row:
            w,h=r['size'];row.update(image=np.full((h,w,3),255,np.uint8),mask=np.zeros((h,w),np.uint8),seed=r['seed'],text='RENDER FAILURE')
    return rows

def gallery(dest,family,real,arms):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    sheet=Image.new('RGB',(1200,690),'#f4f6f8');d=ImageDraw.Draw(sheet);d.text((12,8),family+' | custom + DINO mixture',font=font,fill='#182033')
    for j,title in enumerate(['Original','Custom controller','DINO controller','Mixed controller']):d.text((12+300*j,45),title,font=font,fill='#182033')
    for i in range(6):
        ims=[real[i],*[arms[k][i]['image'] for k in ['custom','dino','mixed']]]
        for j,im in enumerate(ims):
            tile=ImageOps.contain(Image.fromarray(im),(286,78),Image.Resampling.NEAREST);sheet.paste(tile,(300*j+(300-tile.width)//2,82+i*96+(78-tile.height)//2))
    d.text((12,665),'First 6 draws; matched synthetic starting recipes/seeds. Generated strings may differ.',font=small,fill='#465364');sheet.save(dest/'renders.png')

def main():
    ROOT.mkdir(exist_ok=True);model=Mixed();summary={};start=time.time()
    U.dump(ROOT/'protocol.json',dict(method='Equal calibrated custom and DINO v2 appearance scores, with DINO contrast guard and one shared validity penalty.',
        candidate_search='Existing parents, swapped text/background, 24 seeded parameter crossovers, then 24 joint mutations around fit leaders.',
        fit='64 unlabeled fit pixels; 6 matched fresh-string renders per candidate.',
        selection='Union of top4 candidates under each comparator and mixture, plus original parents/neutral. Two separate target-selection batches of32 and12 fresh renders each. Mean score plus0.25 SD. Complete=1 and pooled usable>=.90; each batch usable>=.833.',
        diagnostics='First24 previously exposed test pixels; 24 fresh synthetic samples per arm. Test never selects profiles. Existing parent controllers had different original search budgets.',
        ablation='Also choose custom-only and DINO-only winners from the same validation pool. No claim that component metrics are independent evidence.',
        constraints='Target pixels only; no target strings/masks/boxes, no VLM, no new backbone training. CAPRI workflow is unchanged.',
        code_sha256=U.sha(Path(__file__)),weights=dict(dino=U.sha(WEIGHTS_PATH),mask=U.sha(V3/'mask_training/best.pt'),encoder=U.sha(U.ENCODER))))
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family in FAMILIES:
            dest=ROOT/family;dest.mkdir(exist_ok=True);old=WORK/'controller_mvp_v7_dino'/family
            split=U.read(old/'target_splits.json');split['fit']=split['fit'][:64];split['test']=split['test'][:24];U.dump(dest/'target_splits.json',split)
            hashes=[{r['pixel_sha256'] for r in split[k]} for k in ['fit','selection','test']];assert all(not hashes[i]&hashes[j] for i in range(3) for j in range(i))
            target=model.extract(load_images(split['fit']));validation=[model.extract(load_images(split['selection'][j:j+32])) for j in [0,32]]
            bank=U.read(old/'recipe_banks.json')['hybrid'];recipes=dict(fit=fresh(bank['fit'],6,101000),selection=[fresh(bank['validation'][j],12,102000+j*100) for j in range(2)],test=fresh(bank['test'],24,103000));U.dump(dest/'recipes.json',recipes)
            custom=R.expand(U.read(V4/'collections'/family/'single.json'));dino=R.expand(U.read(old/'selected.json')['controls']);neutral=R.expand({})
            rows=render(pool,recipes['fit'],neutral);objective=Objective(target,model.extract([r['image'] for r in rows]));records=[];seen=set()
            def evaluate(p,phase):
                p=R.expand(p)
                if key(p) in seen:return
                seen.add(key(p));rows=render(pool,recipes['fit'],p);score=objective.score(model.extract([r['image'] for r in rows]),target,rows)
                records.append(dict(index=len(records),phase=phase,controls=p,score=score))
                if len(records)%10==0:
                    print(family,'fit',len(records),flush=True);U.dump(ROOT/'status.json',dict(family=family,phase='fit',candidates=len(records),seconds=time.time()-start))
            for phase,p in candidates(custom,dino):evaluate(p,phase)
            parents=sorted(records,key=lambda r:r['score']['objective'])[:4];rng=np.random.default_rng(101117)
            options=dict(size_absolute=[.45,.6,.75,.9],width_absolute=[.7,.85,1.,1.2],spacing_absolute=[-.2,0.,.2],rotation_absolute=[0.,10.,25.],jitter_absolute=[0.,.1],length=[3,4,5,6,7,8],glyph_outline=[0,1,2],frame_width=[0,1])
            for i in range(24):
                p=copy.deepcopy(parents[i%len(parents)]['controls'])
                for k in rng.choice(list(options),2,replace=False):p[k]=rng.choice(options[k]).item()
                evaluate(p,'joint_mutation')
            U.dump(dest/'fit.json',records)
            shortlist={r['index']:r for r in records if r['phase'] in ['neutral','custom','dino']}
            for metric in ['objective','custom_objective','dino_objective']:
                for r in sorted(records,key=lambda r:r['score'][metric])[:4]:shortlist[r['index']]=r
            selected=[]
            for r in shortlist.values():
                scores=[]
                for j in range(2):
                    rows=render(pool,recipes['selection'][j],r['controls']);scores.append(objective.score(model.extract([x['image'] for x in rows]),validation[j],rows))
                eligible=all(s['complete']==1 and s['usable']>=10/12 for s in scores) and np.mean([s['usable'] for s in scores])>=.90
                robust={k:float(np.mean([s[k] for s in scores])+.25*np.std([s[k] for s in scores])) for k in ['objective','custom_objective','dino_objective']}
                selected.append(dict(index=r['index'],phase=r['phase'],controls=r['controls'],scores=scores,eligible=bool(eligible),robust=robust))
            U.dump(dest/'selection.json',selected);eligible=[r for r in selected if r['eligible']]
            if not eligible:
                winners={a:next(r for r in selected if r['phase']=='neutral') for a in ['mixed','custom_selector','dino_selector']}
            else:winners={a:min(eligible,key=lambda r:r['robust'][k]) for a,k in [('mixed','objective'),('custom_selector','custom_objective'),('dino_selector','dino_objective')]}
            U.dump(dest/'selected.json',dict(winners=winners,passed=bool(eligible)));frozen=U.sha(dest/'selected.json')
            # Only after selection has frozen, access diagnostic images.
            real=load_images(split['test']);test=model.extract(real);arms={};scores={};cache={}
            profiles={'custom':custom,'dino':dino,**{k:r['controls'] for k,r in winners.items()}}
            for arm,p in profiles.items():
                if key(p) not in cache:
                    rows=render(pool,recipes['test'],p);s=objective.score(model.extract([r['image'] for r in rows]),test,rows);cache[key(p)]=(rows,s)
                rows,s=cache[key(p)];save_rows(dest,arm,rows);arms[arm]=rows;scores[arm]=s
            assert U.sha(dest/'selected.json')==frozen;gallery(dest,family,real,arms)
            summary[family]=dict(scores=scores,passed_selection=bool(eligible),selected_index=winners['mixed']['index'],selected_phase=winners['mixed']['phase'],
                selected_sha256=frozen,candidates=len(records),validation_candidates=len(selected),mixed_equals_custom=key(winners['mixed']['controls'])==key(custom),mixed_equals_dino=key(winners['mixed']['controls'])==key(dino))
            U.dump(ROOT/'summary.json',summary);print(family,'DONE', {a:round(s['appearance'],3) for a,s in scores.items()},flush=True)
    U.dump(ROOT/'completion.json',dict(complete=True,seconds=time.time()-start,families=list(summary)))

if __name__=='__main__':main()
