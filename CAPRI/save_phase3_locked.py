from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');local=Path(__file__).resolve().parent;target=root/'runs/style_phase3_color_locked_01'
if target.exists():raise RuntimeError('Preserve historical output')
shutil.copytree(local/'phase3_color_locked_02',target)
for name in ('render_color_locked.py','render_phase3_locked.py'):shutil.copy2(local/name,target/name)
report='''# Phase3 color-locked recipe preview

User explicitly authorized Phase3 generation with colors unchanged.32 renders across8 current sources completed, no rendering failures or baseline fallbacks. Phase1 and prior Phase2 font/effect choices retained. Other recipe controls (height/tilt/spacing,line/dot/curve controls) retained: saved search recipes fororiginal5, frozenrecipeCNN averaged onexistingreferenceimages forother3. No training or new-data evaluation.

Color locks: zero RGBoffsets AND restore exact source palettes/gradientvalues to avoid integer conversion; retain each prior glyphRGB; draw operation colors cyclically from same prior per-kind colors, preventing sharedRNG from inventing new colors. Assertions passed for every output. Pixelpositions, antialiasing and color proportions can change with geometry/density; this is colorparameter preservation, not pixelidentity or fixedcolorhistogram.

First local attempt stopped on palette-equality assertion due to float-to-int conversion even with zerooffset; corrected by exactrestoration, failedattempt preserved locally. Colorlocking renderer is an experiment snapshot, defaulthistoricalrenderer unchanged. Current generation is recipe-based; directCNN/transformer/GAN pixelrefiners not used because their outputs cannot ensure thiscolorlock.

Assistant reviewed all32outputs: City gets thickerlines,Sloth excessiveclutter; fonts/count and knownmisclassifications persist. DarkBay/outlined unchanged because recipeszero. No universalsuccessclaim. User sees before/after galleries bank_1.png andbank_2.png. No newpaper adopted.
'''
(root/'docs/STYLE_PHASE3_COLOR_LOCKED_01.md').write_text(report,encoding='utf-8')
for name,text in {
'STYLE_DECISIONS.md':'D066: User authorized Phase3 generation with colors locked, otherrecipecontrols retained.32renders/8sources completed; exactpalette/gradient restoration, glyphRGB and per-kindoperationcolors locked. No failures. Geometry/density may change pixelcolors through coverage; no pixelidentityclaim. Recipe-based only, no unrestrictedpixelrefiner. Visualoutcome mixed, Slothclutterregression. See STYLE_PHASE3_COLOR_LOCKED_01.md.',
'STYLE_HANDOFF.md':'LATEST USER OVERRIDE: Phase3 trial authorized with NO color changes. Completed32renders/8sources runs/style_phase3_color_locked_01. Fonts/effects retained from directbank; noncolorrecipecontrols retained, colorslocked explicitly. No training or globalpromotion; Slothworseclutter, Citythickerlines. AnyfuturePhase3 must preservecolors; never use unrestrictedpixelrefiners thataltercolor. See report. Experimentalrenderer snapshot only, mainrenderer unchanged.',
'THESIS_PAPERS.md':'Colorlocked Phase3 trial uses existingrecipe methods with original engineeringconstraints; no newpaper.32renders, colorassertionspass but qualitymixed; see STYLE_PHASE3_COLOR_LOCKED_01.md.'
}.items():
    with (root/'docs'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')
