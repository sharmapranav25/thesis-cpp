"""Verified pixel input and labeled render output."""
import hashlib
import numpy as np
from PIL import Image
from . import runtime as U

def load_images(rows):
    images=[]
    for row in rows:
        im=np.asarray(Image.open(row['source']).convert('RGB'))
        if hashlib.sha256(im.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('Target pixels changed')
        images.append(im)
    return images

def save_rows(out,name,rows):
    folder=out/name;folder.mkdir(exist_ok=True);records=[]
    for i,row in enumerate(rows):
        image=folder/f'{i:04d}.png';mask=folder/f'{i:04d}_mask.png'
        Image.fromarray(row['image']).save(image);Image.fromarray(row['mask']).save(mask)
        records.append({k:v for k,v in row.items() if k not in ['image','mask']}|dict(image=str(image.relative_to(out)),mask=str(mask.relative_to(out))))
    U.dump(folder/'labels.json',records)
