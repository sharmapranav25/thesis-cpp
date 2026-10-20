# Background regression audit

Compared32 development samples per family,160total; reproduced every latest Phase3 render pixel-for-pixel. Older-font/newer-font paired renders have identical background drawing operations in all160cases. Font adjustment didnot cause the background change.

Phase2 selectedrecipe offsets alter RGB backgrounds and multiply drawing density: CryptBB lines1.57x,width1.73x,dots2.67x; City lines2.55x,width4.03x,dots.54x; Courier lines4.08x,width.95x,dots1.18x. DarkBay and outlined use zero recipe adjustments, so their background mismatch predates Phase2 and cannot be attributed to this recipe change.

A matched rollback zeroed only backgroundRGB offsets and line/dot/curve controls, retaining Phase2 typography controls and latest Phase3 font/effect/width/height.320 current/rollback renders completed. This is a development ablation, not a new-family evaluation. Cached frozen featureinference remains active; no classifiers trained or bypassed.

Assistant reviewed CryptBB,City,Courier first4samples each. Rollback removes pink/blue/green shifts and some clutter, but City becomes too sparse and CryptBB still misses structured grid patterns. Courier backgrounds lose the artificial green tint. DarkBay/outlined had no recipebackgroundchange. There is no uniformlygood historical background version established by this experiment.

Caveat: renderer shares RNG across background and text. Changing background operation counts advances RNG differently, changing glyphcolors/jitter even though fontconfiguration staysfixed. Therefore this is not an exact glyphpixel-preserving rollback. Next implementation should separate background/text RNG streams and validate typography invariance before choosing background adjustments.

Recommendation: keep latestfontmodels/configuration. Treat Phase2 backgroundrecipecontrols as experimental, replace broad whole-image-loss optimization with separatelyvalidated background structure/color constraints. Fix Phase1 grid/line/texture mapping for families whose baseline wasalreadywrong; do not claim rollbackalone fixesstyle.

Evidence: audit.json, fivefamilycomparison.png files, audit_background_regression.py. Existing histories untouched. No newpaper used. Actual cause is localparameterselection/mapping, not a newpublishedmethod claim.
