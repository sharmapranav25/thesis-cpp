"""Deterministic, independently varied recipes and split-contained sibling pairs."""
import collections
import copy
import string
import numpy as np
from .renderer import R


class Sampler:
    def __init__(self, manifest):
        self.fonts = {f['key']: f for f in manifest['fonts']}
        self.families = {}
        for split in ('train', 'dev', 'calibration', 'test'):
            groups = collections.defaultdict(list)
            for f in manifest['fonts']:
                if f['split'] == split:
                    groups[f['family_group']].append(f)
            self.families[split] = dict(sorted(groups.items()))
        self.effects = list(R.SAMPLED)

    def recipe(self, split, index, attempt=0):
        salt = ('train', 'dev', 'calibration', 'test').index(split)
        group, sibling = divmod(index, 2)
        mode = group % 10
        paired = mode < 3
        seed_index = group * 2 if paired else index
        rng = np.random.default_rng([731, salt, seed_index, attempt])
        font_rng = np.random.default_rng([731, salt, seed_index, 181])
        # Both siblings have the same length; pair groups never cross splits.
        length = 2 + group % 7
        h = int(rng.choice([64, 80, 96, 112, 128]))
        aspect = float(rng.uniform(2.0, 6.0))
        w = int(round(h * aspect))
        families = self.families[split]
        family_names = list(families)
        if not family_names:
            raise ValueError('Empty font split: ' + split)
        family_order=np.random.default_rng([731,salt,17]).permutation(len(family_names))
        family = family_names[int(family_order[(group//7)%len(family_names)])]
        chosen = families[family][int(font_rng.integers(len(families[family])))]
        selected = [chosen] * length
        if font_rng.random() < .2:
            other = family_names[int(font_rng.integers(len(family_names)))]
            second = families[other][int(font_rng.integers(len(families[other])))]
            selected = [chosen if font_rng.random() < .5 else second for _ in range(length)]
            selected[0]=chosen
        charset = str(rng.choice(['digits', 'upper', 'lower', 'mixed']))
        pools = dict(digits=string.digits, upper=string.ascii_uppercase+string.digits,
                     lower=string.ascii_lowercase+string.digits, mixed=string.ascii_letters+string.digits)
        chars = []
        for f in selected:
            allowed = ''.join(c for c in pools[charset] if c in f['chars']) or f['chars']
            chars.append(str(rng.choice(list(allowed))))
        if rng.random() < .15 and all(chars[0] in f['chars'] for f in selected):
            chars = [chars[0]] * length
        bg = [255,255,255] if rng.random() < .3 else rng.integers(0,256,3).tolist()
        multicolour = rng.random() < .45
        colours = []
        for i in range(length):
            c = rng.integers(0,256,3).astype(float)
            lum = np.array([.299,.587,.114])
            if abs((c-np.array(bg)) @ lum) < 65:
                c = c*.25 if np.array(bg) @ lum > 128 else 190+c*.25
            colours.append(c.clip(0,255).astype(int).tolist() if multicolour or not colours else colours[0])
        axes = []
        for f in selected:
            axes.append({a['tag']: float(rng.uniform(a['min'], a['max'])) if rng.random()<.5 else a['default'] for a in f['axes']})
        layout = str(rng.choice(['line', 'curve', 'two_layer'],p=[.6,.2,.2]))
        spec = dict(text=''.join(chars),font_keys=[f['key'] for f in selected],axes=axes,colours=colours,
                    layout=layout,size_fraction=float(rng.uniform(.45,.85)),size_jitter=float(rng.choice([0,.1,.25])),
                    rotation=0. if rng.random()<.3 else float(rng.uniform(5,35)),
                    stretch=float(rng.uniform(.7,1.3)),shear=0. if rng.random()<.35 else float(rng.uniform(-.3,.3)),
                    stroke=int(rng.choice([0,0,0,1])),spacing=float(rng.uniform(-.15,.4)),
                    x_position=float(rng.uniform(.1,.9)),y_position=float(rng.uniform(.1,.9)),
                    jitter=float(rng.choice([0,.03,.08,.12])),curve=float(rng.uniform(.02,.13)) if layout=='curve' else 0.)
        k = int(rng.choice([0,1,2,3,4,5,6],p=[.15,.25,.15,.2,.1,.1,.05]))
        focus = self.effects[group % len(self.effects)]
        names = [focus] if k else []
        for name in rng.permutation(self.effects):
            if len(names)>=k:
                break
            if name in names:
                continue
            exclusive = [('bold','thin','hollow'),('wave','morph','perspective','swirl'),('pixelate','resample')]
            if any(name in g and any(n in g for n in names) for g in exclusive):
                continue
            names.append(str(name))
        effects = [dict(name=n,level=int(rng.integers(1,11))) for n in names]
        pair_kind = None
        if mode == 0:
            pair_kind = 'clean_degraded'
            effects = [] if sibling==0 else (effects or [dict(name='blur',level=4)])
        elif mode == 1:
            pair_kind = 'same_fonts_different_content'
            if sibling:
                original_text=spec['text']
                while spec['text']==original_text:
                    spec['text'] = ''.join(str(rng.choice(list(f['chars']))) for f in selected)
        elif mode == 2:
            pair_kind = 'same_content_different_fonts'
            if sibling:
                eligible = [f for fam in families.values() for f in fam if all(c in f['chars'] for c in spec['text'])
                            and f.get('split_group',f['family_group']) != chosen.get('split_group',family)]
                if eligible:
                    f=eligible[int(rng.integers(len(eligible)))]
                    spec['font_keys']=[f['key']]*length
                    spec['axes']=[{a['tag']:a['default'] for a in f['axes']} for _ in range(length)]
                else:
                    pair_kind='same_content_alternative_unavailable'
        return dict(schema='capri-decoder-recipe-v1',split=split,index=index,attempt=attempt,
                    pair_group=f'{split}/{group}' if paired else f'{split}/single/{index}',pair_kind=pair_kind,
                    render_seed=int(np.random.SeedSequence([731,salt,seed_index,attempt,99]).generate_state(1)[0]),size=[w,h],background=bg,effects=effects,text=spec,
                    requested_count=length,requested_multicolour=bool(multicolour),charset=charset,quota_family=family)
