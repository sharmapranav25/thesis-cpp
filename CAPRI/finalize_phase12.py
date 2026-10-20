from pathlib import Path
import json,sys,subprocess,shutil
root=Path(__file__).parent;release=root/'phase12_release_01';test=root/'phase12_validation_01'
profile=json.loads((release/'profiles/city.json').read_text());bad=test/'overlap_evaluation.json';bad.write_text(json.dumps(dict(partition='evaluation',images=[profile['entries'][0]['reference_path_provenance']])))
r=subprocess.run([sys.executable,'-B',str(release/'pipeline.py'),'audit','--corpora',str(test/'city'),'--evaluation-manifest',str(bad)],capture_output=True,text=True)
assert r.returncode!=0 and 'evaluation images used in style references' in r.stderr
repeat=test/'deterministic_repeat';subprocess.run([sys.executable,'-B',str(release/'pipeline.py'),'generate','--family','city','--out',str(repeat),'--count','42','--seed','5501','--partition','train'],check=True)
a=[json.loads(s)['pixel_sha256'] for s in (repeat/'manifest.jsonl').read_text().splitlines()];b=[json.loads(s)['pixel_sha256'] for s in (test/'city/manifest.jsonl').read_text().splitlines()];assert a==b
(release/'VALIDATION.md').write_text('''# Executed release validation

-336generated samples:42per8families,2per each of21cells. Labels/length/type, renderer glyph counts and individual visibility>=.55 checked. Per-sample retries retained in test evidence.
-Byte-equivalent RGB replay of first historical Phase1+Phase2 render for all8families with the exact original text/font/seed/variant. This verifies packaging preserved baseline behavior.
-Fresh CV and frozen three-class inference on2Cityfitting references matched cached image-level feature decisions. Font inference path ran onCPU. This does not establish classifier/font accuracy.
-Copied package to a different directory, generated21validation samples in diverse mode from an unrelated working directory without old source scripts/assets in its module path.
-Generated6,300text draws across allcells/partitions: label sets disjoint across train/validation/test. Stable text split is not a substitute for pixel audit.
-Audited63train/validation image manifests,CSV labels,image hashes and cross-partition text/pixels. Passed.
-Negative leakage test: deliberately reused style-fit image in an evaluation manifest; audit rejected it as required.
-Deterministic repeat:42Cityimages reproduced identical pixel hashes with same seed and settings.
-No PARSeq optimizer steps, no model retraining, no100kbulk generation. No real style/recognition gate claimed.

Evidence source: sibling phase12_validation_01. Installed release includes validation_summary.json. The test runner uses current local staging paths and is retained in release provenance/test evidence rather than a mandatory runtime dependency.
''',encoding='utf8')
result=json.loads((test/'RESULTS.json').read_text());result.update(evaluation_overlap_rejected=True,deterministic_repeat=True,corpus_audit_samples=63)
(release/'validation_summary.json').write_text(json.dumps(result,indent=2))
subprocess.run([sys.executable,'-B',str(root/'seal_phase12.py')],check=True)
subprocess.run([sys.executable,'-B',str(release/'pipeline.py'),'verify'],check=True)
print('FINALIZED')
