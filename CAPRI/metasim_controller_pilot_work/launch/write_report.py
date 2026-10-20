"""Create the saved report and figures from immutable pilot outputs."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pilot as Q
import pilot_render as P


def load_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    results=Q.read(out/'results.json');diagnostics=Q.read(out/'diagnostics.json')
    groups=results['groups'];simple=groups['simple_initial'];before=groups['specialist_initial'];after=groups['specialist_adjusted']
    status=Q.read(out/'status.json');verification=Q.read(out/'verification.json')
    history=Q.read(out/'search_history.json');controls=Q.read(out/'selected_controls.json')['controls']
    split=Q.read(out/'target_split.json');fonts=diagnostics['original_font_summaries']
    initial=load_rows(out/'specialist_initial/labels.jsonl');adjusted=load_rows(out/'specialist_adjusted/labels.jsonl')
    same_fonts=sum(a['text_spec']['fonts']==b['text_spec']['fonts'] for a,b in zip(initial,adjusted))
    same_glyph_fonts=sum([g['font'] for g in a['glyphs']]==[g['font'] for g in b['glyphs']] for a,b in zip(initial,adjusted))
    assert same_fonts==same_glyph_fonts==96
    verification['same_font_lists']=same_fonts
    verification['same_glyph_font_keys']=same_glyph_fonts
    Q.dump(out/'verification.json',verification)
    env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0),neural_updates=0,
             sources={str(path):Q.sha(path) for path in [Path(__file__),Path(Q.__file__),Path(P.__file__),
                        Path(P.C.__file__),Path(P.R.__file__),Path(P.E.__file__),Path(P.S.__file__)]})
    Q.dump(out/'environment.json',env)
    pct=lambda x,y:100*(x-y)/x
    main_rows=[('Simple hybrid start',simple),('Specialist hybrid start',before),('Specialist with controller',after)]
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    colours=['#777777','#4285a8','#258550'];short=['Simple','Specialist','Adjusted']
    for ax,key,title in [(axes[0],'appearance','Appearance distance'),(axes[1],'audit_sliced_wasserstein','Independent appearance audit')]:
        values=[g[key] for _,g in main_rows]
        bars=ax.bar(short,values,color=colours)
        ax.set_title(title+'\nLower is better',fontsize=11);ax.set_ylim(0,max(values)*1.17)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.3f}',ha='center',va='bottom',fontsize=10)
        ax.spines[['right','top']].set_visible(False)
    x=np.arange(1,len(history)+1);values=np.array([r['objective'] for r in history])
    axes[2].plot(x,values,'.',color='#bbb',label='Candidate')
    axes[2].plot(x,np.minimum.accumulate(values),color='#258550',label='Best fit objective')
    axes[2].set_title('Search on the fit subset',fontsize=11);axes[2].set_xlabel('Controller candidate (24 renders each)')
    axes[2].set_ylabel('Fit objective');axes[2].legend(fontsize=8);axes[2].spines[['right','top']].set_visible(False)
    fig.suptitle('City controller pilot: appearance proxies, not font or OCR accuracy',fontsize=12)
    fig.savefig(out/'metrics.png',dpi=180);plt.close(fig)
    with Image.open(out/'comparison.jpg') as sheet:
        sheet.crop((0,0,sheet.width,42+3*120)).save(out/'preview.jpg',quality=94)
    table='| Starting settings | Appearance distance ↓ | DINO distance ↓ | Statistics distance ↓ | Independent audit ↓ | Usable | Strict readability |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,g in main_rows:
        table+=f"| {name} | {g['appearance']:.5f} | {g['dino']:.5f} | {g['stats']:.5f} | {g['audit_sliced_wasserstein']:.4f} | {100*g['usable']:.1f}% | {100*g['strict']:.1f}% |\n"
    diagnostic_table='| Frozen controls applied | Appearance distance ↓ | Reduction from specialist start | Strict readability | Independent audit ↓ |\n|---|---:|---:|---:|---:|\n'
    for key,name in [('colour_only','Colour only'),('text_only','Text only'),('effects_only','Effects only')]:
        d=diagnostics[key]
        diagnostic_table+=f"| {name} | {d['appearance']:.5f} | {pct(before['appearance'],d['appearance']):.2f}% | {100*d['strict']:.1f}% | {d['audit_sliced_wasserstein']:.4f} |\n"
    diagnostic_table+=f"| All controls | {after['appearance']:.5f} | {pct(before['appearance'],after['appearance']):.2f}% | {100*after['strict']:.1f}% | {after['audit_sliced_wasserstein']:.4f} |\n"
    font_table='| Rendered font attribute | Simple start | Specialist start | Adjusted specialist |\n|---|---:|---:|---:|\n'
    for name,key in [('Weight level','median_weight'),('Width level','median_width'),('X height level','median_x_height'),('Slant level','median_slant_level')]:
        font_table+=f"| Mean image median {name.lower()} (0–10) | "+' | '.join(f'{fonts[k][key]:.3f}' for k in ['simple_initial','specialist_initial','specialist_adjusted'])+' |\n'
    font_table+='| Italic images | '+ ' | '.join(f"{100*fonts[k]['italic_fraction']:.1f}%" for k in ['simple_initial','specialist_initial','specialist_adjusted'])+' |\n'
    ci=results['paired_bootstrap']['after_minus_before_95pct']
    report=f'''# City renderer controller pilot

The controller reduced held-out appearance distance by **{pct(simple['appearance'],after['appearance']):.2f}% versus the simple hybrid starting settings** and **{pct(before['appearance'],after['appearance']):.2f}% versus the specialist starting settings**. This is a useful proof that an inexpensive feedback loop can improve the renderer distribution. It has not established better font accuracy, and the generated images still differ substantially from City.

The main run took {status['seconds']:.1f} seconds ({status['seconds']/60:.2f} minutes) on an RTX A4000. The additional diagnostic renders and verification took {diagnostics['seconds']:.1f} seconds. These times cover execution, not setup or development. No neural weights were trained or updated, and no CAPTCHA solver was trained. The simple and specialist checkpoints and the cached feedback model passed the unchanged-hash check.

{table}
Each generated group contains 96 fresh images, compared with 512 held-out City images. The adjusted specialist and its unadjusted baseline use the same 96 strings and seeds. Every main group has 96 unique images, 96 unique strings and complete glyph labels. Usable and strict readability are renderer checks on known synthetic glyphs; they are not OCR success rates. The full controller passes strict readability for 74/96 images, versus 73/96 for the specialist start and 82/96 for the simple start.

The independent grayscale and edge audit fell by {pct(simple['audit_sliced_wasserstein'],after['audit_sliced_wasserstein']):.2f}% versus the simple start and {pct(before['audit_sliced_wasserstein'],after['audit_sliced_wasserstein']):.2f}% versus the specialist start. It was excluded from controller selection. A paired bootstrap with 200 resamples put the change in the primary appearance distance at {ci[0]:.5f} to {ci[1]:.5f} (95% interval; after minus before). This interval covers sampling variation for one fixed controller and one run, not uncertainty across searches or target families.

![Pilot metrics](metrics.png)

The target pool contained 2,000 City images and 1 exact pixel duplicate. After deduplication, the seeded split used 256 images for fitting, 128 for selection and 512 for the final test; the remaining images were unused. Both starting profiles were built from predictions for the same 256 fitting images. Pixel hashes prove that the three subsets are disjoint. Test scores were computed after saving the selected controls. No real answers or target font files contributed to the objective.

The search used 15 legal renderer controls, eight rounds of eight candidates and 24 common recipes per candidate: **1,536 search renders**. Recipe preparation uses the existing measured-label mapper and performs additional renderer calls; validation, final evaluation and diagnostics also add renders. Thus 1,536 is the search budget, not the total number of renderer calls. The three best fitting candidates were evaluated with 48 fresh recipes against the separate selection subset. Candidates had to stay within five percentage points of neutral usability and strict readability. The third-ranked fitting candidate won selection; one higher-ranked candidate failed readability. The test did not trigger further tuning.

Appearance feedback combined 60% frozen DINO ViT-S/8 feature distance and 40% colour, intensity and projection statistics, using multi-kernel maximum mean discrepancy. The existing DINO model has about 21.7 million parameters, but it was loaded from the local cache and kept frozen. Normalisation and kernel scales came only from the fitting subset. A population search adjusted global settings; it did not learn an image-conditioned controller. This is a small experiment inspired by [Meta-Sim](https://arxiv.org/abs/1904.11621), rather than a reproduction of its neural training method.

The selected controls narrowed text to {controls['width_multiplier']:.3f} times its initial horizontal scale and reduced the nominal size to {controls['size_multiplier']:.3f} times its initial value. They added {controls['spacing_delta']:.3f} spacing and {controls['jitter_delta']:.3f} vertical jitter, changed the variable-weight fraction by {controls['weight_fraction_delta']:+.3f}, lowered requested band strength by two levels, multiplied band slope by {controls['band_slope_multiplier']:.3f} and kept curves with probability {controls['curve_keep_probability']:.3f}. Background colours were blended {100*controls['background_mix']:.1f}% toward a light gray, and foreground colours {100*controls['foreground_mix']:.1f}% toward a dark gray. Rounding selected no font replacement and no additional synthetic bold stroke; the probability of forcing effect palettes to gray was zero.

All 96 paired samples retained their starting font lists and individual glyph font keys. Category, stroke class, x height, slant and italic choices therefore stayed unchanged. Weight and width labels changed because they are measured from the rendered glyphs. A positive variable-weight fraction adjustment does not imply heavier final glyphs when size and width change together.

{font_table}
These are synthetic font measurements, not errors against real City font labels. The typography question remains unresolved: the trial mainly changed geometry within the existing font choices. In the first 12 consecutive test examples, coloured backgrounds, pastel bands, outlines and varied font styles remain conspicuous beside City's gray bands and dark digits. Some adjusted text is narrower, but the overall family resemblance is still weak. No examples were selected for visual quality.

![Held-out references and renderer examples](comparison.jpg)

After the main result was frozen, three additional groups applied only its colour, text or effect controls to the same 96 specialist recipes. These are descriptive checks on the already-scored test subset, with no search, retuning or new selection.

{diagnostic_table}
The text-only group reduced the appearance proxy by {pct(before['appearance'],diagnostics['text_only']['appearance']):.2f}%, while strict readability fell from 73/96 to 67/96. That does not demonstrate an all-font accuracy gain. Renderer operations share a seeded random stream, so changing effect count or level can also change later colour and style draws. The ablations describe which settings were applied; they cannot cleanly partition the gain into independent causes.

The pilot added foreground/background colour hooks and a band-slope hook within its worker processes. These were extensions to the exposed controls; original renderer source files were not edited. Labels and readability were measured after each render. Neutral controls reproduced the original renderer exactly in the saved replay check. A stronger next experiment should improve feedback for typography and family style, and compare it with stronger random search or other controllers at equal renderer budgets. The current result does not justify a large GAN training run yet. CryptBB, DarkBay, real OCR accuracy and labelled real font accuracy remain untested here.

The saved evidence is [results.json](results.json), [diagnostics.json](diagnostics.json), [selected_controls.json](selected_controls.json), [verification.json](verification.json), [protocol.json](protocol.json) and [environment.json](environment.json). Run the experiment with the existing project Python environment and `launch/pilot.py --out <new directory>`. It uses local project code, font assets, cached model predictions and cached DINO weights. Run `launch/diagnostics.py --out <completed directory>` for the fixed-control checks, and `launch/write_report.py --out <completed directory>` to regenerate this report. The fitted controller applies to City and has not been installed as the production default.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(report=str(out/'REPORT.md'),reduction_vs_simple=pct(simple['appearance'],after['appearance']),
                         reduction_vs_specialist=pct(before['appearance'],after['appearance']),
                         unchanged_glyph_fonts=same_glyph_fonts),indent=2))


if __name__=='__main__':main()
