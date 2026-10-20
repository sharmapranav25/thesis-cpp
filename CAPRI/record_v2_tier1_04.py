from pathlib import Path
import sys,json,shutil
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');HERE=Path(__file__).parent;src=HERE/'v2_tier1_04';dst=ROOT/'runs/style_v2_tier1_04'
complete=(src/'status.json').exists()
if complete:shutil.copy2(HERE/'audit_v2_tier1_04.py',src/'audit_v2_tier1_04.py')
shutil.copytree(src,dst,dirs_exist_ok=True)
report='''# V2 Tier1: grid specialist experiment04

Eight grid-specific classifiers share a CNN feature extractor: 1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3 (columns x rows). Each grid head is shared across its own blocks. This is eight heads, not 63 independent neural networks. All heads train jointly from synthetic data. Crop presence targets use visible native-mask support >=3 pixels inside the crop. Tiny local fragments can be ambiguous (straight versus curved particularly); labels describe their originating generator feature, not guaranteed local identifiability.

Each crop's feature probabilities, encoded appearance and coordinates enter a learned pooling layer within its grid. Eight resulting summaries enter learned grid pooling to form global context for presence and full-resolution localization. Grid weights are learned attention, not calibrated confidence or a causal explanation. No family ID is supplied. A complete renderer recipe/style-story interface is not yet integrated.

Matched comparison: one shared local classifier versus eight local classifiers, SAME new pooling, local auxiliary supervision, data, optimizer, 1800 steps and checkpoint-selection rule. Existing03 multiview checkpoint provides a further baseline. Shared-versus-specialist is the direct architectural comparison; original03 also differs in fusion and auxiliary loss. Model parameter counts differ; one training seed is exploratory evidence only.

2400 synthetic training images/six fonts and420 selection images/Anton reused from03. Masks loaded at saved8-bit precision, identical for both04 models (03 trained float compositing masks). Local BCE class weights estimated on training only, capped20; mean each grid separately before averaging eight losses, preventing30-cell grid from dominating solely by cell count. Global loss plus unit-weight auxiliary local loss; select minimum global selection loss, excluding auxiliary. Fixed0.5 thresholds, no failed03 calibration carried forward. All heads verified to receive nonzero gradients; clean-target and global-grid consistency checks pass.

Fresh evaluation: 420 images each at seeds51m(same distribution),52m(heldout fonts only),53m(Bezier curve recipe only). No evaluation-driven checkpoint selection. Shared generator remains a limitation; new seeds are not independent-generator validation. Synthetic-only training; real data inference after freezing only,20previously exposed fit examples across five Tier1 families. Existing CV profile data retained beside learned predictions; CV not refitted in this architectural experiment. Tier1 retains nine font families; no Tier2/3 expansion, PARSeq training or Phase3 revival. Historical/default pipeline preserved. No automatic promotion or real-data gatepass.
'''
if complete:
    results=json.loads((src/'results.json').read_text());report+='\n## Results\n\n| Model | Fresh same macro F1 | Font shift macro F1 | Curve shift macro F1 |\n|---|---:|---:|---:|\n'
    for name,r in results.items():report+='| '+name+' | '+' | '.join(f"{sum(r[p]['f1'])/5:.3f}" for p in ('fresh_same','fresh_font','fresh_recipe'))+' |\n'
    report+='\n| Feature | Shared local F1 | Specialist F1 | Shared local FPR | Specialist FPR |\n|---|---:|---:|---:|---:|\n'
    a=results['shared_local_head']['fresh_same'];b=results['grid_specialists']['fresh_same']
    for i,name in enumerate(('line','curve','dot','rectangle','grain')):
        report+='| '+name+' | '+' | '.join(f'{v:.3f}' for v in (a['f1'][i],b['f1'][i],a['fpr'][i],b['fpr'][i]))+' |\n'
    report+='\nArchitecture audit verifies eight distinct parameter sets, gradient isolation for a selected head,63crop targets,all-clean negatives and bottom-right positional labels. Training additionally checks each head receives a nonzero gradient. Frozen real inference records local probabilities and learned grid weights beside the existing CV profile for each reference; no generated output changed.\n'
    report+='\n## Interpretation and decision\n\nSpecialists versus matched shared local head: fresh same-distribution macroF1 .822 vs .815, font-shift .814 vs .819, recipe-shift .813 vs .825. Original03 scores .831/.817/.807. Separate heads help some dot/curve results but do not consistently improve overall behavior. Same-distribution specialist line/curve false-positive rates remain .433/.452. One run cannot establish statistical superiority; no promotion. This experiment implements the requested architecture and tests the hypothesis, not proof that specialization solves the remaining weaknesses. Next hypothesis should address ambiguous crop labels (e.g. a locally straight fragment of a curve), loss weighting and synthetic diversity before scaling classifiers blindly.\n'
    report+='\nFeature order in full JSON: line,curve,dot,rectangle,grain. Read per-feature precision,recall,FPR,native-maskIoU and each grid classifier scores in results.json. Do not infer real CAPTCHA accuracy from these synthetic metrics.\n'
    message='D081 outcome: grid-specialist experiment04 completed. Eight independently parameterized local heads plus learned hierarchical aggregation implemented; shared-local-head matched control and original03 evaluated. Fresh macroF1 specialists .822/.814/.813 versus shared-local .815/.819/.825 and original03 .831/.817/.807 (same/font/recipe). Mixed, no consistent specialist win, no promotion. Results V2_TIER1_GRID_SPECIALISTS_04.md. Jobs completed; automation remains paused.'
else:message='D081: User authorized grid specialists. Implemented eight per-grid heads with shared encoder, explicit synthetic crop supervision, coordinate/feature-aware within-grid pooling and across-grid pooling. Matched shared-local-head control isolates per-grid specialization from auxiliary/fusion changes. 1800steps each, selection synthetic only; fresh51m/52m/53m evaluation after freeze. Existing five families/nine fonts unchanged. Training active locally; no duplicate job, no pipeline promotion.'
(ROOT/'docs/V2_TIER1_GRID_SPECIALISTS_04.md').write_text(report,encoding='utf8')
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md','V2_TIER1_PLAN.md'):
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
if complete:
    with (ROOT/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nV2Tier1 revision04: original engineering experiment with shared CNN, per-grid MLP classifiers, synthetic crop presence supervision and hierarchical learned pooling. No new external paper directly used or reproduction claimed. Inherits prior03 native-resolution decoder. Exact design, controls and results in V2_TIER1_GRID_SPECIALISTS_04.md and runs/style_v2_tier1_04; findings limited to one procedural generator/training seed.\n')
sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
update('v2_tier1_04_complete' if complete else 'v2_tier1_04_training',message,running=not complete,run=str(dst));print(message)
