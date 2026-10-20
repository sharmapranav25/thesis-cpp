"""Read-only CPU audit: head cost, supervision and constant-predictor references.

Uses training/selection labels only. Does not score held-out sets or change training.
"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import yaml

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
RUN = Path(__file__).resolve().parents[1] / 'run_20261003_01'
sys.path.insert(0, str(PROJECT))
from feature_bank.iter2.model import Iter2Model

torch.set_num_threads(1)
cfg = yaml.safe_load((PROJECT / 'runs/iter2_baseline_15_20261003_04/training_config.yaml').read_text())
model = Iter2Model(**cfg['model'])
total = sum(p.numel() for p in model.parameters())
counts = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
manifest = json.loads((RUN / 'data_manifest.json').read_text())
labels = {}
for split in ['train', 'selection']:
    chunks = [np.load(Path(row['path']) / 'index.npz') for row in manifest['splits'][split]]
    labels[split] = {key: np.concatenate([z[key] for z in chunks]) for key in ['font', 'txt_level', 'txt_present']}
    for z in chunks:
        z.close()

font = {}
for j, name in enumerate(['weight', 'width', 'x_height', 'slant']):
    train = labels['train']['font'][:, j+2]
    selection = labels['selection']['font'][:, j+2]
    candidates = [float(np.mean(np.abs(train[train >= 0] - value) <= 1)) for value in range(11)]
    value = int(np.argmax(candidates))
    font[name] = dict(constant_chosen_on_train=value, train_pm1=candidates[value],
                      selection_pm1=float(np.mean(np.abs(selection[selection >= 0]-value) <= 1)))

tp = labels['train']['txt_level'][:, 4]
fs = labels['train']['font'][:, 5]
cost = dict(total_parameters=total, module_parameters=counts,
    effect_head_with_query=128*128+128+128*11+11+128,
    text_head_with_query=160*160+160+160*11+11+160,
    seven_unsupervised_text_ordinal_channels=7*10*(160+1),
    absent_font_stroke_class_channel=160+1)
cost['seven_unsupervised_text_ordinal_percent'] = 100*cost['seven_unsupervised_text_ordinal_channels']/total
result = dict(cost=cost, font_constant_references=font,
    text_font_slant_level_agreement=float(np.mean(tp == fs)),
    text_slant_presence_font_slant_positive_agreement=float(np.mean(labels['train']['txt_present'][:,4] == (fs > 0))),
    slant_train_positive_counts=dict(text=int((tp > 0).sum()),font=int((fs > 0).sum())),
    font_stroke_train_counts={str(int(v)): int((labels['train']['font'][:,1] == v).sum()) for v in np.unique(labels['train']['font'][:,1])},
    scope='Read-only CPU audit; existing train/selection labels; no training changes or benchmark scoring')
(RUN / 'head_utility_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
