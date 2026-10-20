"""Download the user-selected dataset; extract images only under opaque names."""
from boot import ROOT,U
import hashlib,io,urllib.request,zipfile
import numpy as np
from PIL import Image

def main():
    out=ROOT/'simple_captcha_dataset';out.mkdir(exist_ok=True);archive=out/'download.zip'
    url='https://www.kaggle.com/api/v1/datasets/download/bjosttveit/simple-captcha'
    if not archive.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=90) as response,archive.open('wb') as dest:
            while True:
                block=response.read(1024*1024)
                if not block:break
                dest.write(block)
    images=out/'images';images.mkdir(exist_ok=True);rows={};count=0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            # Only suffix is inspected; target answers and generator files are not read.
            if not info.filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):continue
            pixels=np.asarray(Image.open(io.BytesIO(z.read(info))).convert('RGB'));digest=hashlib.sha256(pixels.tobytes()).hexdigest();count+=1
            path=images/(digest+'.png')
            if not path.exists():Image.fromarray(pixels).save(path)
            rows.setdefault(digest,dict(source=str(path),pixel_sha256=digest))
    U.dump(out/'inventory.json',dict(source='https://www.kaggle.com/datasets/bjosttveit/simple-captcha',download_url=url,archive_sha256=U.sha(archive),image_entries=count,unique_images=len(rows),labels_read=False,rows=list(rows.values())))
    print('Downloaded',count,'images;',len(rows),'unique pixel hashes',flush=True)

if __name__=='__main__':main()
