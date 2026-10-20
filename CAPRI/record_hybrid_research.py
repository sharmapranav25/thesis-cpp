from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');src=Path(__file__).parent/'HYBRID_GRID_RESEARCH.md'
shutil.copy2(src,root/'docs/HYBRID_GRID_RESEARCH.md')
with (root/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\n'+src.read_text())
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nResearch started/completed initial assessment: see HYBRID_GRID_RESEARCH.md. GriddedTNP offers spatial latent-grid encoding; visualConformer is closer to requested bidirectional CNN-transformer interaction. Research references only, no new architecture trained or copied yet. Plan controlled one-way/bidirectional/grid-pooling ablations; do not infer success from papers.\n')
print('Research recorded; no training started.')
