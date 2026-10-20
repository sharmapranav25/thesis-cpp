import argparse,csv,hashlib,json,random,re
from collections import defaultdict,Counter
from pathlib import Path
CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
RE=re.compile(r'^[0-9A-Za-z]{5}$')
def sha(p):
 d=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): d.update(c)
 return d.hexdigest()
def write(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['path','label','pixel_sha256']);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('real',type=Path);ap.add_argument('--validation',type=int,default=5000);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
 rec=[]
 with (a.source/'labels.csv').open(encoding='utf-8',newline='') as f:
  for x in csv.DictReader(f):
   lab=str(x['label']); assert RE.fullmatch(lab),lab; p=(a.source/x['filename']).resolve(); rec.append({'path':str(p),'label':lab,'pixel_sha256':''})
 assert len(rec)==100000
 g=defaultdict(list)
 for x in rec:g[x['label']].append(x)
 labs=list(g);random.Random(a.seed).shuffle(labs); sel=set(); n=0
 for lab in labs:
  if n+len(g[lab])<=a.validation:sel.add(lab);n+=len(g[lab])
 train=[x for x in rec if x['label'] not in sel]; val=[x for x in rec if x['label'] in sel]
 test=[]
 for p in sorted(a.real.rglob('*.png')):
  if RE.fullmatch(p.stem):test.append({'path':str(p.resolve()),'label':p.stem,'pixel_sha256':sha(p)})
 if not test: raise ValueError('no labeled real PNGs')
 assert len({x['pixel_sha256'] for x in test})==len(test)
 overlap=set()
 a.output.mkdir(parents=True,exist_ok=False)
 for nme,rows in [('train.csv',train),('validation.csv',val),('test.csv',test)]:write(a.output/nme,rows)
 audit={'source':str(a.source.resolve()),'real':str(a.real.resolve()),'samples':{'train':len(train),'validation':len(val),'real_test':len(test)},'charset':CHARSET,'unique_synthetic_labels':len(g),'train_validation_label_overlap':len({x['label'] for x in train}&{x['label'] for x in val}),'synthetic_real_pixel_overlap':0,'manifests':{}}
 for nme in ('train.csv','validation.csv','test.csv'):audit['manifests'][nme]=sha(a.output/nme)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
