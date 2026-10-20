"""Reject geometry-changing refinement; preserve the valid baseline, not the candidate."""
import sys
from pathlib import Path
PKG=Path(r'E:\pranav\Final sem\sourcefree_synth\releases\CAPRI_v2');sys.path.insert(0,str(PKG))
import refinement
original=refinement.Refiner.__call__
def checked(self,style,text,font_path,seed,variant=0):
 try:return original(self,style,text,font_path,seed,variant)
 except AssertionError:
  im,mask,recipe=self.base_render(style,text,font_path,seed,variant)
  recipe['refinement']=dict(method='Meta-Sim-inspired',checkpoint_sha256=self.digest,gains=None,fallback=True,fallback_reason='Rejected candidate: glyph geometry/RGB or mask invariant failed',exact_glyph_geometry_and_rgb_preserved=True)
  return im,mask,recipe
refinement.Refiner.__call__=checked
import capri
capri.main()
