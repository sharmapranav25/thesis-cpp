from pathlib import Path
import shutil,datetime,json,hashlib
root=Path(__file__).resolve().parent
source=root/'CAPRI_Thesis_Chapter'
docs=Path('E:/pranav/Final sem/sourcefree_synth/docs')
dest=docs/'CAPRI_THESIS_CHAPTER_20260928'
assert not dest.exists(),'Preserve previous publication; choose a new version.'
shutil.copytree(source,dest)
for p in source.rglob('*'):
 if p.is_file():assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((dest/p.relative_to(source)).read_bytes()).digest()
note='''

## Detailed thesis chapter publication — 2026-09-28

Published [CAPRI chapter](CAPRI_THESIS_CHAPTER_20260928/chapter.md), [readable HTML](CAPRI_THESIS_CHAPTER_20260928/chapter.html), [evidence index](CAPRI_THESIS_CHAPTER_20260928/evidence_index.md) and BibTeX. Approximately12,300words cover V1 resource pilots, three-feature LR/ExtraTrees10x3 classification, feature-pop, CV mapping, font/count/refinement failures, V2 specialists/multiview/multilabel/transformers, joint models, CAPRI fusion and Meta-Sim-inspired adaptation. Includes16primary references and8original figures; copied sources hashed. The chapter preserves failed gates and distinguishes synthetic metrics, real prediction prevalence, distribution metrics and unverified OCR benefit.

Reconciliations: ExtraTrees was actually trained in the initial exploratory run; LR was selected. The6909-image whole-directory inference included200Darkbay evaluation images; they are distribution-exposed. The paper-register introduction saying no paper-supplied weights were used is obsolete after DINO; no runtimeVLM remains accurate. Original records are retained rather than rewritten.

Separate finite dataset recovery: found CryptBB coordinator stopped on PermissionError replacing status.json. All10shards were complete. Prepared recover_capri_bank_v2.py with PID-specific temporary status files and bounded PermissionError retries, preserved prior scripts/logs and partial attempts, checked no existing generation workers, and resumed audit without duplicating completed generation. Courier follows. This does not restart the stopped research automation. Other-project PARSeq processes were observed and left untouched; their outcomes are not established by this chapter.
'''
for n in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (docs/n).open('a',encoding='utf8') as f:f.write(note)
with (docs/'THESIS_PAPERS.md').open('a',encoding='utf8') as f:
 f.write('''

## Consolidated chapter and attribution correction — 2026-09-28

The introductory no-pretrained-paper-weights sentence is historical and superseded: DINO pretrained vision weights were used in later counting/recipe/refinement experiments. They are not VLM weights; classifier/font fitting remained synthetic-only, while optional adapters fit unlabelled real distributions. Gridded/Conformer-inspired joint models were subsequently implemented, superseding their earlier proposed-only entries. See [chapter and complete use/non-reproduction table](CAPRI_THESIS_CHAPTER_20260928/chapter.md) and [BibTeX](CAPRI_THESIS_CHAPTER_20260928/references.bib).

Additional foundational attribution verified while writing: Dalal and Triggs (CVPR2005), Histograms of Oriented Gradients for Human Detection, https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf — descriptor lineage for V1 pooled HOG, not reproduction of human-detection experiments. Geurts, Ernst and Wehenkel (MachineLearning63,2006), Extremely Randomized Trees, https://doi.org/10.1007/s10994-006-6226-1 — scikit-learn ExtraTrees candidates actually trained in V1, with LR selected by the predeclared near-tie rule. Related work Learning to Simulate and Ye etal's CAPTCHA GAN solver are background, not implemented methods; PARSeq is downstream context, not a recognition result demonstrated in this chapter. Detailed primary links and deviations retained in the chapter.
''')
print(json.dumps({'published':str(dest),'files':sum(p.is_file() for p in dest.rglob('*'))}))
