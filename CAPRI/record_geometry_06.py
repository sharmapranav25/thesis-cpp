from pathlib import Path
import sys,json,shutil
HERE=Path(__file__).parent;ROOT=Path('E:/pranav/Final sem/sourcefree_synth');src=HERE/'v2_tier1_06';dst=ROOT/'runs/style_v2_tier1_06';complete=(src/'status.json').exists()
if complete:
    for name in ('visualize_geometry_06.py','audit_tilt_baseline_06.py'):shutil.copy2(HERE/name,src/name)
shutil.copy2(HERE/'audit_geometry_06.py',src/'audit_geometry_06.py');shutil.copytree(src,dst,dirs_exist_ok=True)
report='''# V2 Tier1 06: local stroke direction and neighboring-block relations

User idea: locally straight-looking fragments may form a curve once directions and positions are compared across neighboring blocks. Implemented eight grid geometry heads on shared CNN,63crops (1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3). Each predicts stroke presence, local centroid x/y, axial direction cos2theta/sin2theta, geometric coherence and visible stroke coverage. All local targets come from synthetic visible union of line+curve masks. No local parent-effect curve/line labels. Global five-feature presence and visible segmentation remain supervised.

PCA angle uses native pixel units, not stretched crop coordinates. Doubled angles make0/180degrees equivalent. Coherence combines eigenvalue anisotropy with dominant connected-component fraction. Direction loss applies only on present targets with coherence>=0.6; other local regression targets retain their defined support. This is aggregate orientation, not separation of every stroke/character. Coherence is a geometric training target, NOT calibrated confidence. Multiple crossing strokes and occlusion remain difficult.

84horizontal/vertical neighboring pairs across the eight grids. Edge network receives both predicted geometries, absolute centroid displacement, doubled-angle agreement/change, and distance. It predicts connection probability. Message strength uses predicted stroke presence, coherence and connection; messages update both neighboring node embeddings before within-grid and across-grid pooling. Only predicted geometry enters forward; synthetic masks never enter inference. No family IDs or target generators.

Connection supervision uses shared visible connected-component IDs contributing>=3pixels per block. This is a connectivity proxy: intersections may merge different strokes; a shared component can connect via a route outside this cell pair. It does not establish generator-instance identity, endpoint matching or a continuous fitted curve. There is one message-passing step, not a complete multi-hop contour tracer. A global appearance/spatial path remains, so message usefulness must be measured, not presumed.

Matched controls: geometry_without_neighbors versus geometry_with_neighbors. Identical parameters/initialization, local losses including connection loss, synthetic corpus/seed/minibatches, AdamW settings,1800steps,batch12. Only message aggregation enabled/disabled; control message weights are unused. Checkpoint selection uses synthetic global BCE+dice as before. Frozen03/04 evaluated on the same fresh71m/72m/73m images (420each same,font-shift,Bezier-shift). One training seed/procedural generator, fixed0.5 thresholds. No acceptance or deployment claim. Five real fitcohorts/nine fontfamilies retained; real references inference only with existing CV profiles preserved. No PARSeq, Phase3 or automation restart.

Geometry audit: correct horizontal/vertical axial targets, all-empty handling, disconnected parallel segments stay disconnected, continuous neighbors link, both model initializations identical, message path receives nonzero global gradient. Training masks/geometry saved. Density here means line+curve visible occupancy only; not complete texture/feature-density understanding.
'''
if complete:
    r=json.loads((src/'results.json').read_text());report+='\n## Fresh results\n\n| Model | Same curve F1/FPR | Font curve F1/FPR | Recipe curve F1/FPR |\n|---|---|---|---|\n'
    for name,v in r.items():report+='| '+name+' | '+' | '.join(f"{v[p]['f1'][1]:.3f}/{v[p]['fpr'][1]:.3f}" for p in ('fresh_same','fresh_font','fresh_recipe'))+' |\n'
    report+='\n| Geometry model | Same direction MAE | Same stroke F1 | Same connection F1 |\n|---|---:|---:|---:|\n'
    for name in ('geometry_without_neighbors','geometry_with_neighbors'):
        g=r[name]['fresh_same']['geometry'];report+=f"| {name} | {g['axial_mae_degrees']:.2f} degrees | {g['stroke']['f1'][0]:.3f} | {g['connection']['f1'][0]:.3f} |\n"
    report+='''
## Failed tilt gate: do not promote

Assistant review of geometry_preview.png found nearly horizontal predicted arrows. Quantitative tilt_baseline_audit.json confirms96.98%of predictions on valid same-distribution crops are within10degrees of horizontal, while only37.59%of targets are. Model MAE21.503degrees versus constant-horizontal21.312; font shift21.396 versus21.421; curve-recipe shift29.179 versus28.935. It has not learned useful directional discrimination beyond the horizontal prior. Averaged MAE without this baseline would have been misleading.

Neighbor aggregation yields same-distribution curveF1 .759 versus matched no-neighbor .751, but curve-recipe shift falls .675->.607, and same-distribution rectangleF1 .798->.672. This is not validated cross-block curvature reasoning. Stroke/connection metrics are only partial success against synthetic proxy targets. No replacement/default change.

Next work must fix direction learning before promoting neighbor reasoning: investigate orientation-balanced synthetic supervision, orientation-bin classification rather than conditional mean direction, and preservation of local spatial features instead of averaging all crop positions. These are hypotheses for follow-up, not implemented/validated fixes. The current corpus contains mainly horizontal sine curves, and target direction is multimodal; exact cause of collapse remains unproven. All scripts, training logs, checkpoints, failed targets and preview retained. No new paper directly used.
'''
    message='D083 outcome: cross-block geometry06 implemented/trained; tilt gate FAILED.96.98%valid-crop predictions nearhorizontal; MAE21.50 vs constant21.31. Neighbor curveF1 .751->.759 on same recipe but .675->.607 on curve shift, rectangle regression. No claim of learned tilt reasoning/no promotion. V2_STROKE_RELATIONS_06.md records fullfailure; no active training; automation paused.'
else:message='D083: User authorized cross-block tilt/position reasoning. Implemented8local geometry heads, native axial angles/coherence,84neighbor relations with learned visible-connectivity proxy and gated messages. Train matched with/without neighbor aggregation, identical initialization and data; tests pass. Synthetic only; real inference afterfreeze,5cohorts9fonts. Training active; no pipeline promotion.'
(ROOT/'docs/V2_STROKE_RELATIONS_06.md').write_text(report,encoding='utf8')
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md','V2_TIER1_PLAN.md'):
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
if complete:
    with (ROOT/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nV2Tier1 06: original engineering implementation of local PCA-based axial geometry supervision and neighboring-grid message aggregation from predicted geometry. No external paper directly adopted or reproduction claimed. Exact inputs, connectivity-proxy limitations and matched-control outcomes in V2_STROKE_RELATIONS_06.md and runs/style_v2_tier1_06. Do not cite generic graph/transformer papers as directly used.\n')
sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
update('v2_geometry_complete' if complete else 'v2_geometry_training',message,running=not complete,run=str(dst));print(message)
