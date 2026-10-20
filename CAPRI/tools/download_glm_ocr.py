"""Download a pinned official GLM-OCR snapshot and verify published LFS hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/external/GLM-OCR'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=OUT/'hub_metadata.json'
    if metadata.exists():
        info=json.loads(metadata.read_text())
    else:
        with urllib.request.urlopen('https://huggingface.co/api/models/zai-org/GLM-OCR?blobs=true',timeout=60) as response:info=json.load(response)
        metadata.write_text(json.dumps(info,indent=2))
    revision=info['sha'];records=[]
    for item in info['siblings']:
        name=item['rfilename']
        if '/' in name or not (name.endswith(('.json','.safetensors','.jinja','.model','.txt')) or name in ['README.md','LICENSE']):continue
        path=OUT/name;expected=item.get('lfs',{}).get('sha256')
        if not path.exists():
            print('Downloading',name,item.get('size'),flush=True)
            url=f'https://huggingface.co/zai-org/GLM-OCR/resolve/{revision}/{name}?download=true'
            with urllib.request.urlopen(url,timeout=120) as response,path.with_suffix(path.suffix+'.partial').open('wb') as target:
                while chunk:=response.read(4*1024*1024):target.write(chunk)
            path.with_suffix(path.suffix+'.partial').replace(path)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(4*1024*1024):digest.update(chunk)
        if expected:assert digest.hexdigest()==expected,name
        if item.get('size') is not None:assert path.stat().st_size==item['size'],name
        records.append(dict(file=name,sha256=digest.hexdigest(),bytes=path.stat().st_size,published_lfs_sha256=expected))
    (OUT/'download_receipt.json').write_text(json.dumps(dict(repository='zai-org/GLM-OCR',revision=revision,files=records),indent=2))
    print('Download verified',revision,flush=True)


if __name__=='__main__':main()
