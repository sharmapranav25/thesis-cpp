from pathlib import Path
import json,hashlib,shutil,datetime,re
ROOT=Path(__file__).resolve().parent;src=ROOT/'CAPRI_typography_v3';project=Path('E:/pranav/Final sem/sourcefree_synth');dest=project/'releases/CAPRI_typography_v3';docs=project/'docs'
assert json.loads((src/'VALIDATION.json').read_text())['passed']
for name in ['README.md','LENGTH_RESEARCH.md']:
 p=src/name;p.write_text(re.sub(r'^-(?=\S)','- ',p.read_text(encoding='utf8'),flags=re.M),encoding='utf8')
assert not dest.exists(),'New version required; preserve existing release.'
shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__'))
manifest={str(p.relative_to(dest)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
(dest/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(src/'README.md',docs/'STYLE_PHASE2_BROAD_TYPOGRAPHY_01.md');shutil.copy2(src/'LENGTH_RESEARCH.md',docs/'STYLE_LENGTH_RESEARCH_02.md')
intro='''# Active revision — broad typography and length research, 2026-09-28

This section supersedes older phase/count defaults below. Historical protocols remain unchanged.

1. Phase1 retains CV and frozen learned feature inference. Character-length estimation is now a research priority for Phase1, not an accepted model. Existing failed count/DINO results remain failures; no automatic six-character setting from real evaluation labels.
2. Phase2 now means broad balanced typography on the measured family background, not closest-font matching. Use `releases/CAPRI_typography_v3/generate.py` with adjacent CAPRI_v2. It samples35 audited fonts through uniform shuffled decks;50% mixed-font images in expectation; bounded per-glyph size/width/tilt/stroke/position variation with20% gentle examples. Known synthetic labels and glyph checks retained. Default lengths2–8 until independently validated distribution inference. Read STYLE_PHASE2_BROAD_TYPOGRAPHY_01.md.
3. Optional refinement remains separate. New typography release uses unrefined frozen Phase1 profiles for attribution; existing Meta-Sim adapters are not silently applied to this changed font distribution. Old CAPRI_v2, corpora and running recognisers are preserved.

Validation:1050 images across5families;210 paired replay/background-operation checks; neutral typography exactly matches old renderer in5fixtures. These are integrity checks, not OCR-transfer success. Length research recommends explicit character-instance localisation plus a count baseline, after fixing count-dependent layout shortcuts. See STYLE_LENGTH_RESEARCH_02.md. No new model training or continuous-loop restart.

---

'''
p=docs/'STYLE_PIPELINE_PLAN.md';p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
note='''

## Phase2 broad typography implemented; count research refreshed — 2026-09-28

Reason: user-reported epoch8 syntheticCER3.47% vs realDarkBay82.75%, and actual old font exposure Anton61.99%/Pixelify24.667%/Oswald13.343%, motivate typography-diversity hypothesis. Not proof that fonts are the sole bottleneck. New release CAPRI_typography_v3 preserves CAPRI_v2;35audited font files (12existing+23new pinned Google Fonts), uniform shuffled decks, single/mixed glyph fonts, bounded augmentation and labels/geometry provenance. Actual font exposure after retries is audited; not claimed exactly uniform per accepted glyph. Rejected Tinos acquisition had no eligible TTF at the selected directory; no unsupported asset substituted. Runtime does not use font classifier rankings.

1050smoke renders complete; all image/label/hash/bounds/visibility assertions passed.210background-operation/replay checks and5neutral baseline equivalence checks passed. Assistant DarkBay preview: broad typography visibly increased; some thin/hollow examples remain difficult. No OCR benefit measured. Optional Meta-Sim omitted in new experiment because its adapters fit the old typography policy, preserving an interpretable Phase2 ablation. Old datasets/running recognisers remain unchanged.

Count research: class-agnostic character-instance set prediction + direct-count baseline, synthetic-only supervision, no character identity output. Must repair occupied-width/size/count shortcuts before training. Count accuracy needs independent count-only evaluation annotations; model consistency or family-wide prediction prevalence is insufficient. No count model trained/promoted this turn. Papers and deviations saved in STYLE_LENGTH_RESEARCH_02.md.
'''
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (docs/name).open('a',encoding='utf8') as f:f.write(note)
with (docs/'THESIS_PAPERS.md').open('a',encoding='utf8') as f:
 f.write('''

## Broad typography implementation and length research — 2026-09-28

- Jaderberg, Simonyan, Vedaldi, Zisserman (2014), Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition, https://arxiv.org/abs/1406.2227 . Read synthetic rendering/augmentation section. Direct design inspiration for independent broad font variation in releases/CAPRI_typography_v3; not its generator or published recognition result. Per-character mixing is our augmentation, not an attributed reproduced paper result.
- Baek etal. (CVPR2019), Character Region Awareness for Text Detection, https://arxiv.org/abs/1904.01941 . Read region/affinity method. Proposed character localisation supervision; no implementation/weights used. Original real-data adaptation is not adopted. Word boxes are not character counts.
- Carion etal. (ECCV2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872 . Read matching/object-query method. Proposed12query character/not-character detector; no implementation or claimed counting accuracy yet.
- Graves, Fernández, Gomez, Schmidhuber (ICML2006), Connectionist Temporal Classification, https://www.cs.toronto.edu/~graves/icml_2006.pdf . Read blank/path collapse and alignment. Proposed secondary count-only token baseline; repeated tokens require blank separation. Not implemented.

Detailed candidate comparison, known failed DINO/density evidence, data-shortcut repair and validation boundaries: STYLE_LENGTH_RESEARCH_02.md. Earlier thesis chapter remains a dated snapshot; this update postdates it and does not retroactively change its results.
''')
print(json.dumps(dict(release=str(dest),files=len(manifest),validation=True)))
