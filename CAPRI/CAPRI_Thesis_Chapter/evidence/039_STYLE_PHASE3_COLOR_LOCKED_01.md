# Phase3 color-locked recipe preview

User explicitly authorized Phase3 generation with colors unchanged.32 renders across8 current sources completed, no rendering failures or baseline fallbacks. Phase1 and prior Phase2 font/effect choices retained. Other recipe controls (height/tilt/spacing,line/dot/curve controls) retained: saved search recipes fororiginal5, frozenrecipeCNN averaged onexistingreferenceimages forother3. No training or new-data evaluation.

Color locks: zero RGBoffsets AND restore exact source palettes/gradientvalues to avoid integer conversion; retain each prior glyphRGB; draw operation colors cyclically from same prior per-kind colors, preventing sharedRNG from inventing new colors. Assertions passed for every output. Pixelpositions, antialiasing and color proportions can change with geometry/density; this is colorparameter preservation, not pixelidentity or fixedcolorhistogram.

First local attempt stopped on palette-equality assertion due to float-to-int conversion even with zerooffset; corrected by exactrestoration, failedattempt preserved locally. Colorlocking renderer is an experiment snapshot, defaulthistoricalrenderer unchanged. Current generation is recipe-based; directCNN/transformer/GAN pixelrefiners not used because their outputs cannot ensure thiscolorlock.

Assistant reviewed all32outputs: City gets thickerlines,Sloth excessiveclutter; fonts/count and knownmisclassifications persist. DarkBay/outlined unchanged because recipeszero. No universalsuccessclaim. User sees before/after galleries bank_1.png andbank_2.png. No newpaper adopted.
