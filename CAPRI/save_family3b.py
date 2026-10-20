from pathlib import Path
import json,shutil,sys
import numpy as np
root=Path('E:/pranav/Final sem/sourcefree_synth');here=Path(__file__).parent;src=here/'family_phase3b_01';results=json.loads((src/'results.json').read_text());audit=json.loads((src/'audit.json').read_text());assert json.loads((src/'status.json').read_text())['state']=='complete'
rows=[]
for name,r in results.items():
    a=audit[name];rows.append(f"| {name} | {r['real_fit']}/{r['real_selection']}/{r['real_report']} | {r['steps_trained']} / {r['selected_step']} | {r['report_before']:.3f} -> {r['report_after']:.3f} | {a['dino_before']:.3f} -> {a['dino_after']:.3f} |")
report='''# Own-family Phase3B GAN trial01

User correction implemented: eight separate generators AND eight separate critics, independently initialized. No family conditioning, shared trained refiner weights, cross-family batches, or sequential fine-tuning of another family's model. A generator starts from its own family's Phase2 synthetic image; its critic sees only that family's unlabeled real fit images and refined synthetics. Frozen upstream feature/font infrastructure remains common, as intended. Same architecture/hyperparameters is not shared learned weights. Classifiers remain frozen/synthetic-trained; only these refiners fit unlabeled real images. Count disabled, five generic characters retained.

## Training and data

Larger sources use96 existing fit references:72fit/12selection/12report, repeated fit sampling. Smaller sources use proportional splits; Pymlex only2/1/1 is a smoke test. Two independent synthetic seeds per fit image, one per selection/report image. No real strings/labels used. Source paths/hashes and synthetic texts/seeds/renderer recipes saved per partition. Split hashes are disjoint within this run and fit membership audit passed. All are historically exposed development data; upstream font choices/variants had broader prior exposure. Not unseen evaluation; no historical reserved evaluation reused.

Each family GAN trains at least500 steps, at most1500, batch8 unpaired same-family draws. Evaluate every100 steps; stop after4 checks without improvement, select lowest selection grayscale feature-moment distance. Original unchanged image is eligible (step0). Separate report split never selects checkpoint. Budget cap does not prove convergence; a best checkpoint at1500 is budget-limited. Adam generator0.0005/critic0.0002; mean/std grayscale moments plus0.1 non-saturating local adversarial loss and0.1 L1 self-regularization. Small conv generator, local conv critic,192x64. No recognizer, OCR, VLM or target-specific generator assets. No claim that repeated epochs automatically learn the distribution.

## Direct pixel operation and preservation

Unlike the previous spatial-only pilot, this model predicts an RGB residual directly, not renderer controls. Predicted background RGB is projected to32 representative RGB values from that same synthetic input or the original pixel. Hard projection forward, straight-through residual gradient backward. Colors are exact existing input values; glyph support plus2pixel border copied exactly. Palette consists of16 most frequent colors plus16 lexicographically spread input colors, padded if needed. This is original engineering, not a standard quantizer reproduction. Native synthetic image is resized nearest-neighbor before both baseline and refiner; no RGB introduced by that resizing. No real image patches are copied.

Important limitation: NO NEW RGB VALUES is weaker than unchanged spatial color distribution. Existing colors may cover larger areas, changing perceived background tone/meanRGB. Audit reports this shift explicitly. This implementation cannot change protected glyph shapes/fonts, despite the broader aim of image refinement. Hard masks may leave halos/boundaries. Exact pixels alone do not establish perceptual character preservation or readability, especially for hollow characters.

## Results

| Family | Real fit/select/report | Steps trained / selected | Report moment distance | Report DINO distance |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Moment distance is related to training/selection and can be exploited; frozen DINO CLS was computed only afterward as a diagnostic, not optimized or used for checkpoint selection. Both are proxies, not truth or recognition accuracy. Distinct checkpoint SHA256 for all8 and own-family fit membership verified. Character-mask pixel identity and output RGB membership assertions pass for every report image. See audit.json for actual meanRGB shifts. No default promotion. See VISUAL_REVIEW.md for assistant inspection.

Artifacts: per-family selected.pt, last_generator.pt, last_critic.pt, training.json, splits.json, synthetic records, before/after PNGs and comparison.png; overall protocol/results/status/audit; two full bank galleries. Previous mixed-family direct_phase3b_01 preserved independently. Automation remains paused; this is a bounded requested experiment, not resumed continuous loop.
'''
(src/'REPORT.md').write_text(report,encoding='utf8')
for file in ('run_family3b.py','audit_family3b.py'):shutil.copy2(here/file,src/file)
dst=root/'runs/style_family_phase3b_01';assert not dst.exists();shutil.copytree(src,dst)
(root/'docs/STYLE_FAMILY_PHASE3B_01.md').write_text(report,encoding='utf8')
papers='''
## Direct Phase3B and own-family GAN trials

- Goodfellow et al., Generative Adversarial Networks (2014), https://arxiv.org/abs/1406.2661 . Consulted primary abstract; used generator/discriminator adversarial learning principle, original small conv architecture and non-saturating logistic objective. Not a reproduction of original MLP experiments.
- Shrivastava et al., Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN, CVPR2017), https://arxiv.org/abs/1612.07828 . Re-read primary abstract. Directly used design ideas: synthetic input refined toward unlabeled real pixels, local critic, L1 self-regularization. No history buffer, original architecture/schedule or benchmark reproduction. Own-family independent models, color projection, glyph protection and moment objective are project-specific deviations. Actual implementation runs/style_family_phase3b_01/run_family3b.py. Numeric improvements do not establish visual success; see report/visual review. No claim to reproduce SimGAN quality.
- Jaderberg et al., Spatial Transformer Networks (2015), https://arxiv.org/abs/1506.02025 . Consulted primary abstract for differentiable spatial resampling; previous shared spatial-only direct_phase3b_01 used original dense displacement, hard nearest-color forward and bilinear surrogate backward, not an affine STN reproduction. Shared trial mostly unchanged outputs; abandoned by user in favor of separate family training. New own-family trial uses RGB residual projection, not spatial flow.
- Prior ViT patch-token inspiration applies to shared transformer's original architecture only. DINO is posthoc frozen CLS diagnostic for own-family trial; no DINO training loss, VLM or OCR. Straight-through color projection is original engineering here, not attributed to the above papers as their algorithm.
'''
with (root/'docs/STYLE_PIPELINE_PLAN.md').open('a',encoding='utf8') as f:f.write('\n\nLATEST: optional Phase3B is direct image refinement with a separately trained model per family, using only that family real/synthetic fit pixels and validation checkpoint selection. Shared multi-family refinement does not satisfy current user intent. Phase1/Phase2 baseline remains default; count disabled, automation paused. First own-family GAN trial completed; see STYLE_FAMILY_PHASE3B_01.md for failures and color-coverage limitations. No promotion.\n')
for file,text in {'THESIS_PAPERS.md':papers,'STYLE_DECISIONS.md':'D074 outcome — Eight own-family GANs trained with repeated ownfit sampling, validationselected checkpoints, separate reportimages. Separate models and fitmembership audited. Exactprotected pixels/inputRGB membership pass; perceivedcolor coverage and maskboundary artifacts remain possible. No automaticpromotion from moment/DINO scores. See STYLE_FAMILY_PHASE3B_01.md and run visual review.','STYLE_HANDOFF.md':'LATEST COMPLETE: runs/style_family_phase3b_01 eight independently trained own-family GANs, not sharedconditionalmodel. Per-family logs/checkpoints and reports saved. Read STYLE_FAMILY_PHASE3B_01.md and VISUAL_REVIEW.md. Count remainsdisabled; noactivejob; automationPAUSED. No defaultpromotion.'}.items():
    with (root/'docs'/file).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('family_gan_complete','Eight separate own-family GANs trained and audited; results saved, no automatic promotion. Automation paused.',running=False,run=str(dst))
print('Saved',dst)
