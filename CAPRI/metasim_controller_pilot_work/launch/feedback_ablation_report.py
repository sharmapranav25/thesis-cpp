"""Verify and report the frozen DINO-only and statistics-only experiments."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pilot as Q

NAMES={'simple_initial':'Simple hybrid start','specialist_initial':'Specialist start',
       'mixed_adjusted':'60% DINO + 40% stats','dino_only_adjusted':'100% DINO','stats_only_adjusted':'100% stats'}


def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve();r=Q.read(out/'results.json');g=r['groups']
    protocol=Q.read(out/'protocol.json');parent=Path(protocol['parent']);split=Q.read(out/'target_split.json')
    prior=Q.read(parent/'target_split.json')['splits']
    old_hashes={p['pixel_sha256'] for values in prior.values() for p in values}
    new_hashes={p['pixel_sha256'] for p in split['fresh_evaluation']}
    old_paths={p['source'].lower() for values in prior.values() for p in values}
    new_paths={p['source'].lower() for p in split['fresh_evaluation']}
    assert len(new_hashes)==512 and not old_hashes&new_hashes and not old_paths&new_paths
    verification=Q.read(out/'verification.json');verification['fresh_hash_and_source_paths_disjoint']=True
    base=rows(out/'specialist_initial/labels.jsonl');font_details={}
    for name in NAMES:
        labels=rows(out/name/'labels.jsonl');assert len(labels)==96
        for row in labels:
            with Image.open(out/row['image']) as im: pixels=np.asarray(im.convert('RGB'))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixel_sha256']
        font_details[name]=dict(font_families=dict(Counter(row['glyphs'][0]['family'] for row in labels)),
            stroke_classes=dict(Counter(row['font']['dominant_stroke_class'] for row in labels)),
            glyph_font_keys=list(sorted({s['font'] for row in labels for s in row['glyphs']})),
            mean_pairwise_dino_distance=float(np.mean(Q.distance(np.load(out/f'{name}_features.npz')['dino'],
                                                                 np.load(out/f'{name}_features.npz')['dino']))))
        if name.endswith('adjusted'):
            assert all(a['text']==b['text'] and a['seed']==b['seed'] for a,b in zip(labels,base))
            font_details[name]['changed_font_lists']=sum(a['text_spec']['fonts']!=b['text_spec']['fonts'] for a,b in zip(labels,base))
            font_details[name]['changed_glyph_font_keys']=sum([s['font'] for s in a['glyphs']]!=[s['font'] for s in b['glyphs']] for a,b in zip(labels,base))
    for mode,key in [('dino_only','dino'),('stats_only','stats')]:
        history=Q.read(out/mode/'search_history.json');assert len(history)==64
        for candidate in history:
            assert np.isclose(candidate['appearance'],candidate[key],atol=1e-12,rtol=0)
            assert np.isclose(candidate['objective'],candidate[key]+.8*(1-candidate['usable'])+.15*(1-candidate['strict']),atol=1e-12,rtol=0)
        for choice in Q.read(out/mode/'selection.json'):
            assert np.isclose(choice['score']['appearance'],choice['score'][key],atol=1e-12,rtol=0)
    verification['saved_png_hashes_verified']=480
    verification['feedback_is_exactly_the_named_channel']=True
    verification['selected_file_precedes_evaluation_features']=(out/'selected_all_controls.json').stat().st_mtime < (out/'target_evaluation_features.npz').stat().st_mtime
    assert verification['selected_file_precedes_evaluation_features']
    assert all(Q.sha(path)==sha for path,sha in protocol['source_hashes'].items())
    assert all(Q.sha(parent/path)==sha for path,sha in protocol['parent_hashes'].items())
    Q.dump(out/'verification.json',verification);Q.dump(out/'font_details.json',font_details)
    stat_details={}
    for name in ['real_evaluation']+list(NAMES):
        f=np.load(out/('target_evaluation_features.npz' if name=='real_evaluation' else f'{name}_features.npz'))['stats']
        stat_details[name]=dict(mean_rgb=f[:,:3].mean(0),mean_colour_range=float(f[:,11].mean()),
                               nearly_white_pixel_fraction=float(f[:,12].mean()),dark_pixel_fraction=float(f[:,13].mean()))
    Q.dump(out/'pixel_statistics.json',stat_details)
    reduce=lambda before,after:100*(before-after)/before
    simple=g['simple_initial'];before=g['specialist_initial'];mixed=g['mixed_adjusted'];dino=g['dino_only_adjusted'];stats=g['stats_only_adjusted']
    comparisons={name:{key:reduce(simple[key],value[key]) for key in ['dino','stats','hybrid_reference','audit_sliced_wasserstein']} for name,value in g.items()}
    Q.dump(out/'reductions_vs_simple.json',comparisons)
    table='| Controller settings | DINO MMD² ↓ | Statistics MMD² ↓ | Common 60/40 score ↓ | Independent audit ↓ | Strict readability |\n|---|---:|---:|---:|---:|---:|\n'
    for name,label in NAMES.items():
        v=g[name];table+=f"| {label} | {v['dino']:.5f} | {v['stats']:.5f} | {v['hybrid_reference']:.5f} | {v['audit_sliced_wasserstein']:.4f} | {100*v['strict']:.1f}% |\n"
    font_table='| Synthetic font measurement | Simple start | Specialist start | 60/40 controller | 100% DINO | 100% stats |\n|---|---:|---:|---:|---:|---:|\n'
    for key,label in [('median_weight','Mean image median weight'),('median_width','Mean image median width'),
                      ('median_x_height','Mean image median x height'),('median_slant_level','Mean image median slant')]:
        font_table+='| '+label+' (0–10) | '+' | '.join(f'{g[name][key]:.3f}' for name in NAMES)+' |\n'
    colours=['#777777','#4380a0','#259658','#ba763d','#8063ad']
    fig,axs=plt.subplots(1,3,figsize=(12,3.9),constrained_layout=True)
    ticks=['Simple','Specialist','60/40','DINO','Stats']
    for ax,key,title in zip(axs,['dino','stats','hybrid_reference'],['DINO MMD²','Statistics MMD²','Common 60/40 score']):
        values=[g[name][key] for name in NAMES]
        bars=ax.bar(ticks,values,color=colours)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(values)*1.15);ax.set_title(title+'\nLower is better',fontsize=10)
        ax.tick_params(axis='x',labelrotation=25,labelsize=9);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('City controller feedback: 512 fresh real images and 96 renders per group',fontsize=12)
    fig.savefig(out/'metrics.png',dpi=180);plt.close(fig)
    dino_ci=r['paired_bootstrap']['dino_only_adjusted']['after_minus_specialist_before_95pct']
    stats_ci=r['paired_bootstrap']['stats_only_adjusted']['after_minus_specialist_before_95pct']
    report=f'''# City controller feedback comparison

Both requested searches are complete: 100% DINO appearance feedback and 100% image-statistics feedback. Each used the same starting recipes, seed, 15 renderer controls, 1,536 search renders and readability rules as the earlier 60/40 experiment. The two new searches and shared evaluation took {r['total_seconds']:.1f} seconds ({r['total_seconds']/60:.2f} minutes), excluding setup and report generation. Neural weights stayed frozen.

On 512 fresh City images, DINO-only reduced its DINO discrepancy by **{reduce(simple['dino'],dino['dino']):.2f}% versus the simple hybrid start**, while statistics discrepancy increased **{-reduce(simple['stats'],dino['stats']):.2f}%**. Statistics-only reduced statistics discrepancy by **{reduce(simple['stats'],stats['stats']):.2f}% versus simple**. The previous 60/40 controller still has the lowest common score and independent appearance audit among these five settings in this run. This is a comparison of individual searches at a small fixed budget, not proof of the best feedback weights generally.

{table}
Compare values within each column. The common score applies the original 60/40 weighting to every result for reference; it was not used to select either new controller. The previous controller's numbers differ from the first report because this evaluation uses a different set of real images. All groups contain 96 unique images, 96 unique strings and 96 complete rendered texts, and all pass basic renderer usability. Strict readability is a synthetic glyph guard, not OCR accuracy.

![Feedback comparison](metrics.png)

The DINO controller passes strict readability on 85/96 images, compared with 82/96 for the simple start and 74/96 for both the 60/40 and statistics-only controllers. Its selected controls replace the initial font choices with **Reddit Mono**, reduce nominal font size to 0.802 of its initial value, add vertical jitter of 0.187 and flatten band slope to 0.200 of its initial value. Foreground and background colours are blended about 54% toward gray, and 23% of effects are forced to gray palettes. Statistics-only also selects Reddit Mono, but leaves band slope unchanged and applies little colour adjustment.

In the first 12 consecutive examples, DINO-only gives more consistent digit shapes and visibly staggered vertical placement, which looks closer to the displayed City references than the earlier range of ornate fonts. This is a visual judgment of these examples. It also produces gray backgrounds where City has white space, and retains some colour. At the feedback resize, nearly white pixels occupy {100*stat_details['dino_only_adjusted']['nearly_white_pixel_fraction']:.1f}% of DINO renders versus {100*stat_details['real_evaluation']['nearly_white_pixel_fraction']:.1f}% in the real evaluation images. This brightness mismatch accompanies the worse statistics score; it does not establish which individual statistic caused the increase.

{font_table}
Both new controllers change the glyph font keys in {font_details['dino_only_adjusted']['changed_glyph_font_keys']}/96 paired images. The original 60/40 controller kept those keys unchanged. This makes DINO-only worth examining for typography, but neither DINO distance nor these synthetic font measurements establish real font accuracy. We have no labelled real font errors or downstream recognition scores from this trial.

![City references and controlled renders](comparison.jpg)

The fitting and selection subsets remain the original 256 and 128 City images. Both new policies use fresh independent population searches with identical first-round candidates, then choose among their own top three fitting candidates and neutral on the selection subset. The winner must stay within five percentage points of neutral usability and strict readability. The DINO winner ranked second on fitting; the statistics winner ranked third, with another candidate failing readability. Their saved choices were fixed before evaluation. The fresh 512-image evaluation subset is disjoint by pixel hash and source path from all 896 images used in the earlier pilot. The specialist start and all adjusted variants preserve the same generated strings and seeds; the simple baseline reuses its original outputs.

For DINO-only, the appearance objective is DINO kernel MMD²; for statistics-only, it is statistics kernel MMD². Each retains the original penalties of 0.8 times the unusable fraction and 0.15 times the strict-readability failure fraction. Both feature channels are computed for cross-metric reporting, but the unused channel has exactly zero weight in fitting and selection. Thus the statistics-only runtime includes DINO diagnostics and is not a CPU-only speed benchmark. Statistics normalisation and all kernel bandwidths are fixed from the original fitting subset. The independent grayscale and edge audit contributes to neither search.

With 200 paired bootstrap resamples, the DINO-only change relative to the specialist start has a 95% interval of {dino_ci['dino'][0]:.5f} to {dino_ci['dino'][1]:.5f} for DINO MMD², and {dino_ci['stats'][0]:.5f} to {dino_ci['stats'][1]:.5f} for statistics MMD². Its common score interval spans zero ({dino_ci['hybrid_reference'][0]:.5f} to {dino_ci['hybrid_reference'][1]:.5f}). Statistics-only improves statistics MMD² with an interval of {stats_ci['stats'][0]:.5f} to {stats_ci['stats'][1]:.5f}. These intervals reflect sampling variation for fixed controllers and one search seed; they do not measure repeatability across searches.

The original checkpoints, renderer files, previous selected settings and previous results are unchanged. The new policies are saved as experiment artifacts and were not installed as production defaults. All 480 saved PNG hashes and complete glyph labels were checked. The evidence is [results.json](results.json), [verification.json](verification.json), [font_details.json](font_details.json), [pixel_statistics.json](pixel_statistics.json), [selected controls](selected_all_controls.json) and [protocol.json](protocol.json). Reproduce both searches with `launch/feedback_ablation.py --parent <original pilot directory> --out <new output directory>` using the existing project Python environment.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    Q.dump(out/'completion_manifest.json',dict(status='complete',report=str(out/'REPORT.md'),comparison=str(out/'comparison.jpg'),
        original_sources_unchanged=True,neural_updates=0,search_renders=3072,production_default_changed=False,
        code_hashes={str(path):Q.sha(path) for path in [Path(__file__),Path(__file__).with_name('feedback_ablation.py')]}))
    print(json.dumps(dict(reductions_vs_simple=comparisons,pixel_statistics=stat_details,font_changes=font_details),indent=2,default=Q.default))


if __name__=='__main__':main()
