"""Render the simple hybrid's descriptions before manual target adjustment."""
import copy
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image

PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(PROJECT))
from feature_bank.iter2 import controller as C
from feature_bank.iter2.mapping import map_recipe
from captcha_effects.captcha import render_captcha

OUT=Path(__file__).resolve().parent
OLD=PROJECT/'runs/inference_dataset_v1_20261003'


def main():
    assert json.loads((OUT/'inference_report.json').read_text())['state']=='complete'
    predictions={str(Path(r['source']).resolve()).lower():r for r in
        map(json.loads,(OUT/'predictions.jsonl').read_text().splitlines())}
    result={}
    for family,seed in [('CryptBB',202),('DarkBay',304),('City',410)]:
        ref=json.loads((OLD/family.lower()/'manifest.jsonl').read_text().splitlines()[0])['real']
        row=predictions[str(Path(ref).resolve()).lower()]
        content=C.CONTENT[int(np.argmax([row['text_features'][h]['probability'] for h in C.CONTENT_HEADS]))]
        description=copy.deepcopy(dict(size=row['size'],length=row['length'],font=row['font'],
            effects=row['effects'],text_features=row['text_features'],content=content))
        for name,g in description['text_features'].items():
            if name in C.BINARY:
                g['level']=None
        es,ts=C.initial_recipe(description,np.random.default_rng(seed))
        mapped=map_recipe(C.bank(),tuple(description['size']),es,ts,description,seed,passes=1)
        rendered=mapped['after']
        if rendered is None:
            raise RuntimeError(f'{family}: model starting recipe did not render')
        Image.open(ref).convert('RGB').save(OUT/f'{family}_target.png')
        Image.fromarray(rendered['image']).save(OUT/f'{family}_before.png')
        record=dict(family=family,seed=seed,source_index=row['index'],source=row['source'],
            prediction=row,description=description,effects_spec=mapped['effects_spec'],text_spec=mapped['text_spec'],
            mapping_history=mapped['history'],mapping_calls=mapped['calls'],
            mapping_distance_before=mapped['before_distance'],mapping_distance_after=mapped['after_distance'],
            legibility=rendered['text']['legibility'],actual_text_labels=rendered['text'],
            actual_effect_labels=rendered['effects'])
        (OUT/f'{family}_starting_point.json').write_text(json.dumps(record,indent=2))
        result[family]=dict(length=row['length'],font=row['font'],content=content,
            effects={h:dict(probability=round(v['probability'],3),level=v['level'])
                     for h,v in row['effects'].items() if v['present']},
            text_features={h:dict(probability=round(v['probability'],3),level=v['level'])
                     for h,v in row['text_features'].items() if v['present']},
            chosen_fonts=[C.bank().by_key[key]['family'] for key in mapped['text_spec']['fonts']],
            mapping_calls=mapped['calls'],legibility=record['legibility'])
    (OUT/'starting_point_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
