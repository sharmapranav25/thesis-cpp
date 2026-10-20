"""Encode assistant visual review of the ten prediction-blind contact sheets.

These rules serialize a human-readable review record, not runtime classifiers.
They must never be used as training labels or as audited evaluation labels.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import FEATURES,read,write,digest

root=Path(__file__).resolve().parents[1]/'runs/feature_bank_01'
manifest=read(root/'manifest.json')
draft_path=root/'draft_annotations.json'
if draft_path.exists(): raise FileExistsError('Preserve prior drafts; use a new revision')
grid={3,6,8,13,14,15,19,23,25,32,34,35,36}
blocks={4,5,7,9,10,11,12,16,17,21,24,29,33,38,39}
curves=set(range(1,41))-grid-blocks
outlined_straight={1,2,9,11,12,13,15,16,17,19,20,24,26,27,28,29,30,32,33,34,36,37,40}
outlined_curved={3,4,5,6,7,8,10,14,18,21,22,23,25,31,35,38,39}
rows=[]
for r in manifest['images']:
    n=int(r['id'].rsplit('-',1)[1]); labels={f:'absent' for f in FEATURES}
    if r['dataset']=='cryptbb':
        labels['dot_scatter']='present'
        labels['nonlinear_text_deformation']='uncertain'
        labels['hollow_lettering']='uncertain'; labels['glyph_border']='uncertain'; labels['glyph_shadow']='uncertain'
        if n in grid:
            labels['periodic_grid']='present'; labels['straight_interference']='present'
            note='Visible repeated colored grid and fine speckles. Typography has varying decorative contours; border/hollow/shadow and intrinsic font versus deformation remain uncertain.'
        elif n in blocks:
            labels['background_blocks']='present'; labels['decorative_shapes']='uncertain'
            note='Visible colored rectangular patches and fine speckles. Whether patches should also count as decorative shapes is ambiguous. Typography attribution uncertain.'
        else:
            labels['curved_interference']='present'; labels['straight_interference']='uncertain'; labels['background_texture']='uncertain'
            note='Dense colored curving strokes and speckles. Individual straight strokes and background texture cannot be separated confidently in this view; typography attribution uncertain.'
    elif r['dataset']=='city':
        labels['straight_interference']='present'
        note='Multiple pale gray approximately straight bands behind dark glyphs. Spacing and directions are irregular, so no periodic-grid label. No visible nonlinear bending beyond glyph design/rotation.'
    elif r['dataset']=='courier':
        labels['curved_interference']='present'; labels['straight_interference']='uncertain'
        labels['dot_scatter']='uncertain'; labels['nonlinear_text_deformation']='uncertain'
        labels['glyph_border']='uncertain'; labels['glyph_shadow']='uncertain'
        note='Colored wavy interference is visible. Fine speckles can be stroke/glyph breakup rather than separate dot scatter. Bending, borders and shadow-like edges cannot reliably be attributed at this scale.'
    elif r['dataset']=='darkbay':
        labels['background_blocks']='present'; labels['decorative_shapes']='present'
        labels['straight_interference']='uncertain'; labels['periodic_grid']='uncertain'
        note='Tonal rectangular patches and outlined rectangular decoration. Shared rectangle edges make separate interference-line and periodic-grid interpretations ambiguous. Filled glyphs are mainly rotated rather than visibly bent.'
    else:
        labels['background_gradient']='present'; labels['hollow_lettering']='present'
        labels['straight_interference']='present' if n in outlined_straight else 'uncertain'
        labels['curved_interference']='present' if n in outlined_curved else 'uncertain'
        note='Visible left-to-right gray gradient, hollow letter strokes and a thick crossing stroke. Crossing stroke includes shallow curvature/straight segments; its dominant category is a draft judgment. Hollow strokes are not labeled as a border on filled glyphs.'
    rows.append(dict(id=r['id'],image=r['image'],labels=labels,notes=note,reviewed=False))
write(draft_path,dict(version='assistant-draft-v1',manifest_sha256=digest(root/'manifest.json'),
    reviewer='assistant visual review',role='assistant',status='UNAUDITED',
    review_evidence='All ten *_1.png/*_2.png contact sheets opened before classifier development; no predictions used. Small-view ambiguity retained. No target transcription.',
    images=rows))
print('Wrote 200 UNAUDITED drafts, 2400 feature judgments')
