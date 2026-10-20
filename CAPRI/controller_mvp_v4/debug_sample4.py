from setup4 import ROOT,U,V1,FAMILIES,CHARSET
import copy
import numpy as np
from corpus4 import C,compatible_font

def main():
    C.install();rng=np.random.default_rng(42019);used=set();fonts=[f['key'] for f in U.read(U.WORK/'learned_controller_20261004/fonts.json')]
    for i in range(13):
        while True:
            text=''.join(rng.choice(list(CHARSET),size=int(rng.integers(2,9))))
            if text not in used:used.add(text);break
        seed=int(rng.integers(1000000,2**30));font=fonts[int(rng.integers(len(fonts)))]
    family='SimpleCaptcha';recipe=U.read(V1/(family+'_run_01')/'fit_recipes.json')[1]
    policies=dict(hybrid=C.expand(C.R.NEUTRAL),single=U.read(ROOT/'collections'/family/'single.json'),diverse=U.read(ROOT/'collections'/family/'archive.json')[1])
    for name,p in policies.items():
        r=copy.deepcopy(recipe);r['seed']=seed;r['text_spec'].update(text=text,charset='mixed',layout='line',curve=0.);r['description']['content']='mixed';r['text_spec']['fonts']=[compatible_font(f) for f in r['text_spec']['fonts']]
        width,height=r['size'];r['size']=[max(128,min(512,round(width*64/height))),64]
        r['text_spec']['bold_stroke']=int(r['text_spec'].get('bold_stroke',0))+1
        p=copy.deepcopy(p);p['length']=None
        if p['font']:p['font']=compatible_font(p['font'])
        row=C.render((r,p));print(name,'text',text,'complete',row['complete'],'usable',row['usable'],'guard',row['guard'],'effects_spec',row['effects_spec'],'text_spec',row['text_spec'])

if __name__=='__main__':main()
