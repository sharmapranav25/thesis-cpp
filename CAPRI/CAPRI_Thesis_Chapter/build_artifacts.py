"""Freeze documentary evidence and prepare the portable chapter bundle."""
from pathlib import Path
import json, hashlib, shutil, re, datetime

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'evidence').mkdir(exist_ok=True)
(OUT/'assets').mkdir(exist_ok=True)
groups={
'1–3: scope, provenance and evaluation': [PROJECT/'docs'/n for n in ['FEATURE_BANK_CLASSIFICATION_PLAN.md','STYLE_PIPELINE_PLAN.md','DATASETS.md','VALIDATION.md','STYLE_DECISIONS.md','STYLE_HANDOFF.md','THESIS_PAPERS.md']],
'4: resource pilots and the first 10 by 3 corpus':[PROJECT/'docs/feature_bank'/n for n in ['RESOURCE_ACQUISITION_01.md','ADAPTERS.md','PILOT_01.md','MAPPING_COVERAGE.md','REGIONS_02.md','LABELED_BRICKS_03.md','CORPUS_CANDIDATE_01.md']],
'5: V1 classical classifiers and gates':[WORK/'feature_bank_work/docs/feature_bank/EXPLORATORY_01.md',*[WORK/'feature_bank_work/runs/feature_bank_exploratory_01'/n for n in ['protocol.json','results.json','descriptor_schema.json','status.json']],*[PROJECT/'docs'/n for n in ['STYLE_FROZEN_FIVE_01.md','STYLE_FEATURE_AUDIT_01.md','STYLE_FEATURE_NEGATIVES_01.md','STYLE_STRAIGHT_REVISION_EVAL_01.md','STYLE_PHASE1_01_REVIEW.md']]],
'6: backgrounds, fonts, count and unsuccessful refinement':[PROJECT/'docs'/n for n in ['STYLE_BACKGROUND_REGRESSION_01.md','STYLE_BACKGROUND_BLOCKS_FIX_01.md','STYLE_PHASE2_PILOT_01.md','STYLE_PHASE2_PROTOCOL_01.md',*[f'STYLE_PHASE3_FONT_0{i}.md' for i in range(1,6)],'STYLE_CHARACTER_COUNT_01.md','STYLE_COUNT_HEATMAPS_01.md','STYLE_DINO_RECIPE_01.md','STYLE_PHASE3_PARAMETER_AUDIT_01.md','STYLE_PHASE3_CLUTTER_ABLATION_01.md','STYLE_PHASE3_COLOR_LOCKED_01.md']],
'6 continued: own-family pixel refiners':[WORK/'family_phase3b_01'/n for n in ['protocol.json','results.json','audit.json','VISUAL_REVIEW.md']],
'7–8: multiview and specialists':[PROJECT/'docs'/n for n in ['V2_TIER1_INITIAL_RESULTS.md','V2_TIER1_REVISION03.md','V2_TIER1_GRID_SPECIALISTS_04.md','V2_CURVE_DEBUG_05.md','V2_STROKE_RELATIONS_06.md','V2_FEATURE_SPECIALIST_PLAN.md','V2_CURVE_FAILURE_DIAGNOSIS.md','V2_DOT_SPECIALIST_PLAN.md','V2_DOT_SPECIALIST_RESULTS.md','V2_TWO_STAGE_STROKES.md','V2_STROKE_GRID_10X3.md']],
'8 continued: full real inference':[WORK/'FULL_DATASET_DOT_STATUS.md',WORK/'curve_diagnosis.json'],
'9–10: multilabel and joint inference':[PROJECT/'docs'/n for n in ['V2_MULTILABEL_PLAN.md','V2_MULTILABEL_RESULTS.md','V2_MULTILABEL_REAL_PHASE12.md','V2_TRANSFORMER_MULTILABEL_PLAN.md','V2_TRANSFORMER_MULTILABEL_RESULTS.md','V2_TRANSFORMER_REAL_PHASE12.md','V2_JOINT_MAPPING_PLAN.md','V2_JOINT_GRIDDED_RESULTS.md','V2_JOINT_CONFORMER_RESULTS.md','HYBRID_GRID_RESEARCH.md']],
'11–12: CAPRI and Meta-Sim-inspired fitting':[WORK/'CAPRI_v2'/n for n in ['README.md','NEXT_SESSION.md','requirements.txt','INTEGRITY.json']],
'12 continued: refinement numerical evidence':[WORK/'capri_metasim_01'/n for n in ['PLAN.md','RESULTS.md','protocol.json','results.json','INTEGRITY.json']],
'13: corpus production and recovery':[WORK/n for n in ['generate_darkbay_100k.py','capri_safe_generation.py','recover_capri_bank.py','cryptbb_recovery.log','capri_recovery.stderr.log']],
}
manifest=[];missing=[]
lines=['# Evidence index','',f'Frozen at {now}. Paths refer to the original machine; local evidence copies remain readable when this folder is moved. Hashes verify copied bytes, not scientific validity.','', 'Chapter section numbers map to the groups below. Historical documents may contain superseded statements; inspect their run scope and later corrections.','']
for group,paths in groups.items():
 lines+=['## '+group,'']
 for p in paths:
  if not p.exists():
   missing.append(str(p));continue
  b=p.read_bytes();name=f'{len(manifest)+1:03d}_{p.name}';dest=OUT/'evidence'/name;dest.write_bytes(b)
  manifest.append(dict(section=group,source=str(p),copy='evidence/'+name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
  lines.append(f'- [{p.name}](evidence/{name}) — original: `{p}`')
 lines.append('')

# Find exact variant names instead of silently inventing missing source links.
if missing:
 lines+=['## Source-name mismatches to resolve','']+[f'- `{p}`' for p in missing]

figs=[('feature_bank_work/runs/feature_bank_feature_pop_01/comparison.png','feature_pop.png'),('family_phase3b_01/bank_1.png','rejected_family_gan.png')]
grid=PROJECT/'runs/feature_bank_regions_02/cryptbb/overview.png'
if grid.exists():figs.append((str(grid),'v1_grid.png'))
for fam in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 figs.append((f'capri_five_character_gallery/{fam}_real.png',fam+'_comparison.png'))
for src,name in figs:
 p=Path(src);p=p if p.is_absolute() else WORK/p
 shutil.copy2(p,OUT/'assets'/name)
 manifest.append(dict(section='figures',source=str(p),copy='assets/'+name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

status={}
for fam in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 d=Path('E:/pranav/Final sem/synth_data')/(fam+'_capri_v2_100k')
 row={}
 for n in ['status.json','audit.json']:
  p=d/n
  if p.exists():row[n]=json.loads(p.read_text())
 row['shards']={p.parent.name:json.loads(p.read_text()) for p in (d/'shards').glob('*/status.json')}
 status[fam]=row
(OUT/'dataset_status_snapshot.json').write_text(json.dumps(dict(time=now,families=status),indent=2),encoding='utf8')
(OUT/'evidence_manifest.json').write_text(json.dumps(dict(time=now,sources=manifest,missing=missing),indent=2),encoding='utf8')
(OUT/'evidence_index.md').write_text('\n'.join(lines),encoding='utf8')

chapter=(OUT/'chapter.md').read_text(encoding='utf8')
appendix='''

## Appendix B. Recorded visual evidence

These are existing experiment exports, not newly selected best outputs. The three final examples per family use the earlier fixed five-character protocol: numeric, alphabetic and alphanumeric; first three fit recipes in their saved order, seeds 63000–63002. Real references are untranscribed development examples, not paired reconstructions. The CAPRI/refined columns preserve the same chosen synthetic text and font within each pair. They illustrate the single-font Meta-Sim trial, not a completed evaluation of the later font-mixture corpus.

### B.1 Original grid and feature-pop representations

![Original V1 grid demonstration](assets/v1_grid.png)

Figure B1. The recorded V1 grid view. Overlap preserves nearby pixels but does not establish object continuity or feature accuracy.

![Feature-pop transformations](assets/feature_pop.png)

Figure B2. Original RGB, CLAHE, feature pop, Canny and morphological views. Character strokes are strongly enhanced alongside interference; this is why feature pop cannot be treated as a feature-only mask.

### B.2 Rejected direct image refinement

![Own-family GAN trial with visible halos](assets/rejected_family_gan.png)

Figure B3. Own-family GAN trial. The protected text survives, but surrounding halos, contours and colour coverage change. Baseline errors also remain. This experiment was rejected; it is included to show a failure mechanism, not a successful refinement result.

### B.3 CAPRI and bounded Meta-Sim-inspired recipe refinement
'''
for fam in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 appendix+=f'\n![{fam}: real reference, CAPRI and recipe refinement](assets/{fam}_comparison.png)\n\nFigure B{4+len(re.findall("Figure B[4-9]",appendix))}. {fam.replace("_"," ")}: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.\n'
if '## Appendix B.' not in chapter:
 chapter+=appendix
(OUT/'chapter.md').write_text(chapter,encoding='utf8')
refs=re.findall(r'^\*\*\[R(\d+)\]\*\* (.+)$',chapter,re.M)
authors={
'1':'Wang, Zhangyang and Yang, Jianchao and Jin, Hailin and Shechtman, Eli and Agarwala, Aseem and Brandt, Jonathan and Huang, Thomas S.',
'2':'Shrivastava, Ashish and Pfister, Tomas and Tuzel, Oncel and Susskind, Josh and Wang, Wenda and Webb, Russ',
'3':'Sun, Baochen and Saenko, Kate',
'4':'Dosovitskiy, Alexey and others',
'5':'Ashman, Matthew and Diaconu, Cristiana and Langezaal, Eric and Weller, Adrian and Turner, Richard E.',
'6':'Peng, Zhiliang and Huang, Wei and Gu, Shanzhi and Xie, Lingxi and Wang, Yaowei and Jiao, Jianbin and Ye, Qixiang',
'7':'Caron, Mathilde and Touvron, Hugo and Misra, Ishan and Jégou, Hervé and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand',
'8':'Lempitsky, Victor and Zisserman, Andrew',
'9':'Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr',
'10':'Kar, Amlan and Prakash, Aayush and Liu, Ming-Yu and Cameracci, Eric and Yuan, Justin and Rusiniak, Matt and Acuna, David and Torralba, Antonio and Fidler, Sanja',
'11':'Ruiz, Nataniel and Schulter, Samuel and Chandraker, Manmohan',
'12':'Ye, Guixin and others',
'13':'Bautista, Darwin and Atienza, Rowel',
'14':'Epshtein, Boris and Ofek, Eyal and Wexler, Yonatan',
'15':'Dalal, Navneet and Triggs, Bill',
'16':'Geurts, Pierre and Ernst, Damien and Wehenkel, Louis'}
venues={'1':'ACM Multimedia','2':'CVPR','4':'ICLR','5':'ICML, Proceedings of Machine Learning Research 267, 1722--1761','6':'ICCV, 367--376','7':'ICCV','8':'Advances in Neural Information Processing Systems','9':'ICLR','10':'ICCV','11':'ICLR','12':'ACM CCS','13':'ECCV','14':'CVPR','15':'CVPR','16':'Machine Learning, 63, 3--42'}
fields=[]
for num,text in refs:
 title=re.search(r'\*([^*]+)\*',text).group(1)
 year=re.search(r'\((\d{4})\)',text).group(1)
 author=authors[num]
 url=re.findall(r'\]\((https?://[^)]+)\)',text)[-1]
 venue=venues.get(num,'Preprint')
 typ='article' if num=='16' else ('misc' if num=='3' else 'inproceedings')
 venuekey='journal' if num=='16' else ('howpublished' if num=='3' else 'booktitle')
 fields.append(f'@{typ}{{R{num},\n  title = {{{title}}},\n  author = {{{author}}},\n  year = {{{year}}},\n  {venuekey} = {{{venue}}},\n  url = {{{url}}}\n}}')
(OUT/'references.bib').write_text('\n\n'.join(fields)+'\n',encoding='utf8')
checks=dict(word_count=len(chapter.split()),reference_count=len(refs),sections=len(re.findall(r'^## ',chapter,re.M)),copied_sources=len(manifest),missing_sources=missing,generated=now)
(OUT/'build_report.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print(json.dumps(checks,indent=2))
