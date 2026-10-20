from pathlib import Path
import json, re, hashlib
p=Path(__file__).resolve().parent
m=json.loads((p/'evidence_manifest.json').read_text())
for row in m['sources']:
 assert hashlib.sha256((p/row['copy']).read_bytes()).hexdigest()==row['sha256'],row['copy']
assert not m['missing']
chapter=(p/'chapter.md').read_text(encoding='utf8')
assert set(re.findall(r'\[R(\d+)\]',chapter))==set(map(str,range(1,17)))
for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)',chapter):assert (p/link).is_file(),link
meta=json.loads(next((p/'evidence').glob('*results.json')).read_text()) # Schema checks below use exact source identities.
def load(suffix):
 row=next(r for r in m['sources'] if r['source'].replace('\\','/').endswith(suffix))
 return json.loads((p/row['copy']).read_text())
refine=load('capri_metasim_01/results.json')
for fam,score in [('cryptbb',3.79),('city',4.33),('courier',7.20),('darkbay',27.71),('outlined_text_captcha',5.96)]:
 assert round(100*refine[fam]['relative_mmd_reduction'],2)==score
lr=load('feature_bank_exploratory_01/results.json')
for f,score in [('straight_interference',.8832),('curved_interference',.8800),('dot_scatter',.9455)]:
 assert round(lr[f]['algorithms']['LR']['metrics']['test']['ap'],4)==score
audit=load('family_phase3b_01/audit.json')
assert sum(v['dino_after']>v['dino_before'] for v in audit.values())==7
report={'passed':True,'source_hashes_verified':len(m['sources']),'references':16,'figures':len(re.findall(r'!\[',chapter)),'numeric_checks':['V1 LR test AP','five-family Meta-Sim relative improvements','own-family GAN DINO degradation count'],'limits':'These checks validate selected transcriptions and source integrity, not every scientific interpretation or claim.'}
(p/'evidence_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
