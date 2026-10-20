"""Post hoc controls ablations and artifact verification; no search or selection."""
import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pilot as Q
import pilot_render as P


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def font_summary(values):
    result = {'categories': dict(Counter(r['font']['dominant_category'] for r in values)),
              'stroke_classes': dict(Counter(r['font']['dominant_stroke_class'] for r in values)),
              'italic_fraction': float(np.mean([r['font']['italic_any'] for r in values]))}
    for key in ['median_weight', 'median_width', 'median_x_height', 'median_slant_level']:
        result[key] = float(np.mean([r['font'][key] for r in values]))
    for key in ['min_glyph_height', 'min_final_stroke', 'min_glyph_contrast']:
        result[key] = float(np.mean([r['guard'][key] for r in values]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    start = time.perf_counter()
    locked = Q.read(out/'selected_controls.json')
    selected_hash = Q.sha(out/'selected_controls.json')
    main_results_hash = Q.sha(out/'results.json')
    selected = np.array(locked['choice']['vector'])
    ablations = {}
    for name, indices in [('colour_only', list(range(5))),
                          ('text_only', list(range(5,12))),
                          ('effects_only', list(range(12,15)))]:
        v = P.NEUTRAL.copy()
        v[indices] = selected[indices]
        ablations[name] = v
    Q.dump(out/'diagnostic_protocol.json', dict(
        purpose='Post hoc decomposition of the locked controller; no tuning, no new selection',
        selected_sha256=selected_hash, main_results_sha256=main_results_hash,
        controls={name: dict(zip(P.NAMES,v)) for name,v in ablations.items()},
        renders_per_group=96, fitting=False, selection=False,
        limitation='Descriptive ablations on the already-scored test subset, not new confirmatory tests'))

    split = Q.read(out/'target_split.json')['splits']
    hashes = {name: {r['pixel_sha256'] for r in values} for name,values in split.items()}
    assert all(len(hashes[name])==len(values) for name,values in split.items())
    assert not (hashes['fit']&hashes['selection'] or hashes['fit']&hashes['test'] or hashes['selection']&hashes['test'])
    actual_targets = Q.real_images(split['test'])
    for im,r in zip(actual_targets,split['test']):
        assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
    base_groups = {name: rows(out/name/'labels.jsonl') for name in
                   ['simple_initial','specialist_initial','specialist_adjusted']}
    verification = dict(pixel_split_disjoint=True, held_out_target_hashes_verified=512,
                        neutral_replay=Q.read(out/'neutral_replay_check.json'), group_checks={})
    for name,values in base_groups.items():
        assert len(values)==96
        for r in values:
            im = np.asarray(Image.open(out/r['image']).convert('RGB'))
            assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
            assert r['text']==r['text_spec']['text']
            assert len(r['glyphs'])==len(r['text']) and all(g['box'] is not None for g in r['glyphs'])
        verification['group_checks'][name] = dict(images=96, pixel_hashes_verified=True,
            unique_pixels=len({r['pixel_sha256'] for r in values}),
            unique_strings=len({r['text'] for r in values}), complete_text=True)
    before,after = base_groups['specialist_initial'],base_groups['specialist_adjusted']
    assert all(a['text']==b['text'] and a['seed']==b['seed'] for a,b in zip(before,after))
    verification['specialist_pairs_same_strings_and_seeds']=96
    protocol = Q.read(out/'protocol.json')
    verification['source_hashes_unchanged'] = all(Q.sha(path)==value for path,value in protocol['source_hashes'].items())
    assert verification['source_hashes_unchanged']
    events = rows(out/'events.jsonl')
    assert next(r['seconds'] for r in events if r['stage']=='selecting_on_separate_target_subset') < next(r['seconds'] for r in events if r['stage']=='final_held_out_evaluation')
    verification['selected_file_timestamp_before_test_stage'] = (out/'selected_controls.json').stat().st_mtime < (out/'target_test_features.npz').stat().st_mtime
    Q.dump(out/'verification.json',verification)

    scaling = np.load(out/'feedback_scaling.npz')
    feedback = object.__new__(Q.Feedback)
    feedback.mean=scaling['stats_mean'];feedback.std=scaling['stats_std']
    feedback.audit_mean=scaling['audit_mean'];feedback.audit_std=scaling['audit_std']
    feedback.directions=scaling['directions'];feedback.widths=Q.read(out/'kernel_bandwidths.json')
    real = dict(np.load(out/'target_test_features.npz'))
    fonts = [r['key'] for r in Q.read(out/'font_candidates.json')]
    recipes = []
    for row in before:
        with Image.open(out/row['image']) as im: size=list(im.size)
        recipes.append(dict(size=size,seed=row['seed'],index=row['index'],
            effects_spec=row['effects_spec'],text_spec=row['text_spec']))
    Q.dump(out/'final_specialist_recipes.json',recipes)
    model = Q.Features()
    result = {}
    generated_groups = {}
    with ProcessPoolExecutor(max_workers=6,initializer=P.install) as pool:
        replay = Q.generated(pool,[recipes[0]],P.NEUTRAL,fonts)[0]
        assert replay['pixel_sha256']==before[0]['pixel_sha256']
        for name,vector in ablations.items():
            rendered = Q.generated(pool,recipes,vector,fonts)
            assert all(r['text']==b['text'] for r,b in zip(rendered,before))
            f = Q.features(model,rendered)
            score = feedback.score(f,real,rendered)
            score['audit_sliced_wasserstein']=feedback.audit(f,real)
            score['images']=len(rendered)
            score['font_summary']=font_summary(rendered)
            result[name]=score
            generated_groups[name]=rendered
            Q.save_group(out,name,rendered)
            np.savez(out/f'{name}_features.npz',**f)
            print(json.dumps(dict(group=name,seconds=round(time.perf_counter()-start,1),**score)),flush=True)
    result['original_font_summaries'] = {name:font_summary(values) for name,values in base_groups.items()}
    result['selected_unchanged'] = Q.sha(out/'selected_controls.json')==selected_hash
    result['main_results_unchanged'] = Q.sha(out/'results.json')==main_results_hash
    assert result['selected_unchanged'] and result['main_results_unchanged']
    result['seconds']=time.perf_counter()-start
    Q.dump(out/'diagnostics.json',result)
    sheet = Image.new('RGB',(1440,42+8*120),'white')
    draw=ImageDraw.Draw(sheet)
    names=['Held-out City references','Colour controls only','Text controls only','Effects controls only']
    for j,name in enumerate(names): draw.text((j*360+8,12),name,fill='black')
    for i in range(8):
        ims=[actual_targets[i]]+[generated_groups[name][i]['image'] for name in ablations]
        for j,pixels in enumerate(ims):
            im=Image.fromarray(pixels);im.thumbnail((348,90));x=j*360+6;y=42+i*120
            sheet.paste(im,(x,y+(90-im.height)//2))
            draw.text((x,y+94),'reference (unpaired)' if j==0 else f'{i}: {before[i]["text"]}',fill='black')
    sheet.save(out/'diagnostic_comparison.jpg',quality=94)
    print('Diagnostics and verification complete',flush=True)


if __name__=='__main__': main()
