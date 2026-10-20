from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');local=Path(__file__).resolve().parent
target=root/'runs/style_direct_bank_01'
if target.exists():raise RuntimeError('Preserve existing output')
shutil.copytree(local/'direct_bank_renders_01',target)
shutil.copy2(local/'render_direct_bank.py',target/'render_direct_bank.py')
report='''# Direct Phase1 plus fonts: bank preview01

Eight current sources, four renders each (32 total): CryptBB, City, Courier, DarkBay, outlined, roncuevas, pymlex, sloth. Fresh CV measurements and frozen font05 inference; historical frozen feature predictions reused with image hash verification. Font selection uses mean posterior over up to16 existing fit/reference images per source, hollow only at mean>=.7. No real training, no old recipe/refiner or typography gains, no grayscale/line-cap experiments. Generic5characters retained. This is a development preview, not unseen evaluation or matched comparison with past galleries.

Font choice is new direct inference, not the old font/width/height search that depended on old Phase2 recipes. All32 before/after font background-operation lists match exactly. Assistant inspected all previews: family color differences remain but layout/noise/line mistakes persist, outlined and pymlex choose inappropriate pixel font, sloth falsehollow, count/alphabet unresolved. No phasepass. Do not claim all old typography improvements retained; this direct-inference baseline still needs font search validated in the new order. Optional Phase3 remains deferred for user assessment.
'''
(root/'docs/STYLE_DIRECT_BANK_01.md').write_text(report,encoding='utf-8')
for name,entry in {
'STYLE_DECISIONS.md':'D065: User requested all bank renders under revised order. Completed32 renders across8 sources, freshCV+frozenfont05 inference, cached image-hash-verified feature inference. No formerPhase2 recipes. Mean-font posterior selection, not inherited recipe-dependent search. Backgroundoperations unchanged32/32. Existingfit/reference only, no newunseenclaim. Wrongpixel/hollow/count/layout remain; no gatepass. See STYLE_DIRECT_BANK_01.md.',
'STYLE_HANDOFF.md':'LATEST: All8source directPhase1+fonts gallery complete runs/style_direct_bank_01,32renders, no oldrecipe/refiner. Showed bank_1.png and bank_2.png inchat. Fontselection freshmeanposterior (not oldrecipe-dependent fontsearch); typography needs new-order validation. Phase3 stilldeferred. Noactivejob. See STYLE_DIRECT_BANK_01.md.',
'THESIS_PAPERS.md':'Direct Phase1+fonts bank preview01 uses existing documented methods; no newpaper. Eight sources/32renders; backgroundoperations preserved but severalfont/layout errors persist. See STYLE_DIRECT_BANK_01.md.'
}.items():
    with (root/'docs'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')
