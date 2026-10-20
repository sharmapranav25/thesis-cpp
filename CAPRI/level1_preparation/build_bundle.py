"""Assemble the requested baseline model folders without editing upstream code."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

MODELS = ['PARSeq', 'ViTSTR', 'SATRN', 'TrOCR', 'SVTR', 'SVTRv2', 'CRNN']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('destination', type=Path)
    root = p.parse_args().destination.resolve()
    source = Path(__file__).parent
    shared = root / 'shared'
    shared.mkdir(exist_ok=True)
    for name in ['adapters.py', 'runner.py']:
        shutil.copy2(source / name, shared / name)
    charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    upstream = root / 'upstream'
    provenance = {}
    for name in ['parseq', 'OpenOCR', 'SATRN', 'mmocr']:
        repo = upstream / name
        provenance[name] = {
            'url': subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'], text=True).strip(),
            'commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
        }
    # Vendor only the SATRN components. Keep original notices and computation;
    # replace framework registry/data plumbing with small local constructors.
    vendor = shared / 'satrn_vendor'
    vendor.mkdir(exist_ok=True)
    (vendor / '__init__.py').write_text('', encoding='utf-8')
    mm = upstream / 'mmocr' / 'mmocr' / 'models'
    files = {
        'transformer_module.py': 'common/modules/transformer_module.py',
        'transformer_layers.py': 'common/layers/transformer_layers.py',
        'satrn_layers.py': 'textrecog/layers/satrn_layers.py',
        'shallow_cnn.py': 'textrecog/backbones/shallow_cnn.py',
        'satrn_encoder.py': 'textrecog/encoders/satrn_encoder.py',
        'nrtr_decoder.py': 'textrecog/decoders/nrtr_decoder.py',
    }
    hashes = {}
    for target, rel in files.items():
        src = mm / rel
        text = src.read_text(encoding='utf-8')
        hashes[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
        text = text.replace('from mmocr.registry import MODELS\n','')
        text = text.replace('@MODELS.register_module()\n','')
        text = text.replace('from mmocr.structures import TextRecogDataSample', 'from typing import Any as TextRecogDataSample')
        text = text.replace('from mmocr.models.common.modules import', 'from .transformer_module import')
        text = text.replace('from mmocr.models.textrecog.layers import', 'from .satrn_layers import')
        text = text.replace('from mmocr.models.common import MultiHeadAttention','from .transformer_module import MultiHeadAttention')
        text = text.replace('from mmocr.models.common import PositionalEncoding, TFDecoderLayer', 'from .transformer_module import PositionalEncoding\nfrom .transformer_layers import TFDecoderLayer')
        text = text.replace('from mmocr.models.common.dictionary import Dictionary', 'from .base import Dictionary')
        if 'MODELS.build(act_cfg)' in text:
            text = 'from mmcv.cnn import build_activation_layer\n' + text.replace('MODELS.build(act_cfg)', 'build_activation_layer(act_cfg)')
        text = '# Adapted for standalone baseline training: imports and registry plumbing only.\n' + text
        (vendor / target).write_text(text, encoding='utf-8')
    (vendor / 'base.py').write_text('''from mmengine.model import BaseModule
class BaseEncoder(BaseModule):
    pass
class Dictionary:
    def __init__(self, charset):
        self.num_classes = len(charset) + 3
        self.start_idx = len(charset)
        self.end_idx = len(charset) + 1
        self.padding_idx = len(charset) + 2
class BaseDecoder(BaseModule):
    def __init__(self, dictionary, module_loss=None, postprocessor=None, max_seq_len=30, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if module_loss is not None or postprocessor is not None:
            raise ValueError('Standalone adapter supplies loss and decoding explicitly')
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
''', encoding='utf-8')
    shutil.copy2(upstream / 'mmocr' / 'LICENSE', vendor / 'LICENSE')
    (vendor / 'PROVENANCE.json').write_text(json.dumps({'upstream':provenance['mmocr'], 'source_sha256':hashes, 'modifications':'Imports/registration and base dictionary plumbing adapted; attention/backbone/encoder/decoder computation retained.'},indent=2),encoding='utf-8')
    for model in MODELS:
        folder = root / model
        folder.mkdir(exist_ok=True)
        for sub in ['checkpoints','logs','data']:
            (folder/sub).mkdir(exist_ok=True)
        size = [32,128]
        batch = 16
        variant = model
        if model == 'SATRN':
            size, batch, variant = [32,100], 4, 'MMOCR SATRN shallow (512-d, 12 encoder / 6 decoder layers)'
        elif model == 'TrOCR':
            size, batch, variant = [384,384], 2, 'TrOCR-small architecture, character-level vocabulary, random initialization'
        elif model == 'ViTSTR':
            variant = 'ViTSTR-small, PARSeq rectangular 4x8-patch reproduction'
        elif model == 'SVTR':
            variant = 'OpenOCR SVTR-base CTC reproduction'
        elif model == 'SVTRv2':
            variant = 'OpenOCR SVTRv2 CTC (base dimensions)'
        cfg = dict(model=model, variant=variant, charset=charset, max_label_length=12,
                   image_size=size, batch_size=batch, effective_batch_size=32,
                   epochs=20, learning_rate=0.0003, weight_decay=0.01,
                   warmup_fraction=0.1, gradient_clip=5.0, seed=42,
                   num_workers=0, precision='fp32', initialization='random',
                   train_manifest=None, val_manifest=None,
                   normalization='RGB / 255 then (x - 0.5) / 0.5', augmentation='none')
        if model in ['SVTR','SVTRv2']:
            import yaml
            rel = 'svtrs/svtr_base_ctc.yml' if model == 'SVTR' else 'svtrv2/svtrv2_ctc.yml'
            cfg['architecture'] = yaml.safe_load((upstream/'OpenOCR/configs/rec'/rel).read_text(encoding='utf-8'))['Architecture']
        config = folder/'config.json'
        if not config.exists():
            config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        (folder/'train.py').write_text('''from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'shared'))
from runner import main
if __name__ == '__main__':
    main(Path(__file__).resolve().parent / 'config.json')
''',encoding='utf-8')
        (folder/'train.ps1').write_text('''$ErrorActionPreference = 'Stop'
$python = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\\Scripts\\python.exe'
& $python (Join-Path $PSScriptRoot 'train.py') @args
exit $LASTEXITCODE
''',encoding='utf-8')
        (folder/'data/manifest.example.csv').write_text('path,label\nimages/example.png,aB3x9\n',encoding='utf-8')
        (folder/'README.md').write_text(f'''# {model} — level 1 standard baseline

Variant: {variant}.

Random initialization; no pretrained weights or previous experiment checkpoints.
Case-sensitive 62-character alphabet; editable maximum length 12.
Native model loss, ordinary supervised learning, no curriculum, teacher or target-specific adaptation.
Input size: {size}. Default microbatch: {batch}; effective batch: 32 through accumulation.

From PowerShell in this folder:
```powershell
.\\train.ps1 --smoke
.\\train.ps1 --train-manifest 'D:\\dataset\\train.csv' --val-manifest 'D:\\dataset\\val.csv'
```
CSV headers must be `path,label`. Relative image paths resolve beside the CSV.
Labels remain exact: no case folding or trimming. Missing images, unsupported characters,
overlength strings and train/validation pixel overlap are rejected before training.
See `config.json` for editable settings. `--epochs`, `--batch-size`, `--device cpu` and
`--run-name NAME` are supported. Model weights are created when a run begins, not downloaded.
Outputs go under `checkpoints/NAME` and `logs/NAME`; an existing run is never overwritten.
Best checkpoint uses validation exact-string accuracy, with CER as a secondary metric.
`--evaluate CHECKPOINT --val-manifest PATH` evaluates a saved model.
The `--smoke` command uses a disposable two-image fixture, checks an optimizer update and
inference, and discards weights. It is not dataset training or an accuracy benchmark.

Shared runtime/code live one folder above; keep this folder with `shared`, `upstream`, and `.venv`.
See the root README and provenance/licenses before redistributing.
''',encoding='utf-8')
    (root/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    shutil.copy2(source/'README.md',root/'README.md')
    shutil.copy2(source/'requirements.txt',root/'requirements.txt')
    print('Prepared: ' + ', '.join(MODELS))

if __name__ == '__main__':
    main()
