"""Acquire the official visual-only DINOv2 source and public S/14-register weights."""
import os
import urllib.request
import zipfile
from pathlib import Path
from .common import ROOT, read, write, sha, now

REVISION = '85a24602099d397264d5b30461ad7f3bfd726ca1'
WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth'


def acquire(run):
    root = Path(run)/'pretrained/dinov2'
    root.mkdir(parents=True, exist_ok=True)
    sources = [('source-'+REVISION+'.zip', 'https://codeload.github.com/facebookresearch/dinov2/zip/'+REVISION),
               ('dinov2_vits14_reg4_pretrain.pth', WEIGHTS_URL)]
    records = []
    for name, url in sources:
        destination = root/name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as f:
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    f.write(block)
                f.flush(); os.fsync(f.fileno())
            temporary.rename(destination)
        records.append(dict(path=str(destination.resolve()),url=url,sha256=sha(destination),bytes=destination.stat().st_size))
    source = ROOT/'src/capri_controller/vendor'/('dinov2-'+REVISION)
    with zipfile.ZipFile(root/('source-'+REVISION+'.zip')) as z:
        prefix = 'dinov2-'+REVISION+'/'
        for info in z.infolist():
            if not info.filename.startswith(prefix): raise ValueError('Unexpected archive root')
            relative = info.filename[len(prefix):]
            target = (source/relative).resolve()
            if not target.is_relative_to(source.resolve()): raise ValueError('Unsafe archive path')
            if info.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                payload=z.read(info)
                if target.exists() and target.read_bytes()!=payload: raise ValueError('Existing source differs')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    result=dict(created_at=now(),revision=REVISION,variant='dinov2_vits14_reg',assets=records,
                source=str(source),license=sha(source/'LICENSE'),strict_load_verified=False)
    write(root/'provenance.json',result)
    config=read(ROOT/'configs/local.json')
    config.update(decoder_backbone='dinov2_vits14_reg',decoder_dinov2_source=str(source),
                  decoder_dinov2_checkpoint=str((root/sources[1][0]).resolve()))
    write(ROOT/'configs/local.json',config)
    return result
