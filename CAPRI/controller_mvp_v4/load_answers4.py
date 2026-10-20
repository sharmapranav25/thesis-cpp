"""Evaluation-only filename/CSV annotations joined to frozen images by decoded-pixel hash."""
from setup4 import ROOT,V3,U,FAMILIES
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,re,zipfile
import numpy as np
from PIL import Image

def pixel_hash(source):return hashlib.sha256(np.asarray(Image.open(source).convert('RGB')).tobytes()).hexdigest()

def main():
    dest=ROOT/'evaluation';freeze=U.read(dest/'inference_complete.json')
    assert freeze['answers_read'] is False and freeze['models_frozen_sha256']==U.sha(ROOT/'solvers/frozen_models.json')
    for name,digest in freeze['prediction_sha256'].items():assert U.sha(dest/name)==digest
    maps={f:{} for f in FAMILIES};conflicts={f:set() for f in FAMILIES};sources={}
    def add(f,h,label):
        if h in maps[f] and maps[f][h]!=label:conflicts[f].add(h)
        else:maps[f][h]=label
    archives={
        'SimpleCaptcha':(V3/'simple_captcha_dataset/download.zip',r'[A-Z]{5}','https://www.kaggle.com/datasets/bjosttveit/simple-captcha'),
        'Outlined':(Path('E:/pranav/Final sem/sourcefree_synth/data/external/outlined_text_captcha/source/train.zip'),r'[abcdefghkmnprvwxy23456789]{5}','https://huggingface.co/datasets/muddokon/outlined-text-captcha')}
    for f,(path,pattern,url) in archives.items():
        irregular=[]
        with zipfile.ZipFile(path) as z:
            members=[n for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg'))]
            for name in members:
                label=Path(name).stem
                if not re.fullmatch(pattern,label):irregular.append(name);continue
                add(f,pixel_hash(io.BytesIO(z.read(name))),label)
        sources[f]=dict(kind='Original dataset filename stems; matched by decoded RGB pixel SHA256, never opaque working filenames.',archive=str(path),archive_sha256=U.sha(path),images=len(members),publisher=url,
            skipped_irregular_filename_annotations=irregular,verification='Accepted member stems match the publisher-described character alphabet and length; irregular stems are excluded before scoring. Original filename annotations are accepted as supplied, not independently re-annotated by a human or OCR model.')
    dark=Path('E:/pranav/DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled')
    for p in dark.iterdir():
        if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg'] and re.fullmatch(r'[A-Z0-9]{6}',p.stem):add('DarkBay',pixel_hash(p),p.stem)
    sources['DarkBay']=dict(kind='Original labeled directory filename stems, documented in local DATASETS.md; pixel-hash match to reserved pool images.',directory=str(dark),documentation='E:/pranav/cryptbb_sourcefree/DATASETS.md',unique_labeled_pixels=len(maps['DarkBay']))
    crypt=Path('E:/pranav/model 1/data/eval');csvpath=crypt/'labels.csv'
    with csvpath.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            p=crypt/row['filename']
            if p.exists():add('CryptBB',pixel_hash(p),row['label'])
    sources['CryptBB']=dict(kind='Existing labeled evaluation CSV; only exact decoded-pixel matches to the preregistered diagnostic images are accepted.',csv=str(csvpath),csv_sha256=U.sha(csvpath),unique_labeled_pixels=len(maps['CryptBB']))
    sources['City']=dict(kind='No verified annotation source found. Numeric-looking filenames and screenshot timestamps are not interpreted as answers.',documentation='E:/pranav/cryptbb_sourcefree/DATASETS.md')
    answers={};stats={}
    for f in FAMILIES:
        wanted=U.read(dest/(f+'_predictions.json'))['pixel_sha256']
        answers[f]={h:maps[f][h] for h in wanted if h in maps[f] and h not in conflicts[f]}
        stats[f]=dict(requested=len(wanted),matched=len(answers[f]),ambiguous=sum(h in conflicts[f] for h in wanted),missing=len(wanted)-len(answers[f]))
    U.dump(dest/'answers.json',answers)
    U.dump(dest/'answer_provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inference_completed_utc=freeze['created_utc'],inference_complete_sha256=U.sha(dest/'inference_complete.json'),sources=sources,counts=stats,
        answer_access_boundary=U.read(dest/'answer_access_boundary.json'),restriction='Evaluation only. No model, policy, sampling, training, checkpoint or prediction changes after answer access. No VLM or OCR annotation.'))
    print(stats)

if __name__=='__main__':main()
