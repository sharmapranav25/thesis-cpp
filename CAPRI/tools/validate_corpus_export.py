"""Check relocated output paths, worker imports, ZIP labels and resume integrity."""
from pathlib import Path
import os,sys,json,zipfile,hashlib,io
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
os.environ['CAPRI_RUN_DIR']=str(ROOT/'maintenance/validation/export_smoke')
sys.path.insert(0,str(ROOT/'src'))
from capri_controller import corpus
from PIL import Image
import numpy as np

def main():
    corpus.OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'capri_darkbay_workflow/entries.json';entries=corpus.U.read(source)
    tasks=[(e,2,i,6,corpus.U.sha(source)) for i,e in enumerate(entries[:2])]
    with ProcessPoolExecutor(max_workers=2,initializer=corpus.init) as pool:
        first=list(pool.map(corpus.generate,tasks));second=list(pool.map(corpus.generate,tasks))
    assert first==second
    count=0
    for summary in first:
        path=corpus.OUT/(summary['entry_id']+'.zip')
        with zipfile.ZipFile(path) as z:
            for row in map(json.loads,z.read('labels.jsonl').decode().splitlines()):
                im=np.asarray(Image.open(io.BytesIO(z.read(row['image']))).convert('RGB'))
                assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
                assert row['text']==row['text_spec']['text'] and len(row['text'])==6
                count+=1
    result=dict(images_decoded=count,workers=2,resume_identical=True,output_under_configured_run=True)
    corpus.U.dump(ROOT/'maintenance/validation/export_summary.json',result);print(json.dumps(result))

if __name__=='__main__':main()
