# Synthetic-only character-count pilot01

Task: predict count2-8 only. No OCR, alphabet prediction, VLM, real labels or real training. Original small CNN trained locally onRTX A4000. Independent licensed12fontbank: first10training, last2withheldfonttest.1792training images;448selection,448calibration,448ordinarytest,448withheldfonttest. Random text,length,position,spacing,rotation,outline,noise,lines,blur and polarity. Seedsdisjoint; shortstrings may overlap acrosspartitions, explicitlynotclaimed globallydisjoint. Whole-image gray+edge encoding; no croppedfontviews thatwouldlosecharacters.

1200steps,batch64,AdamW; bestcheckpoint500 selected by minimumsyntheticselectioncrossentropy. Calibrationtemperature1.45; exploratory confidencecutoff.5 chosen onsyntheticcalibration to reach90%empiricalprecision with>=30support. In-sample calibration is not realconfidenceguarantee.

| Test | Exactcount | MAE | Confidentsubset accuracy/coverage |
|---|---:|---:|---:|
| Ordinary synthetic448 |89.29%|.161|92.74%,413/448|
| Withheldfonts448 |93.53%|.085|95.54%,426/448|

Hardercounts6-8 produce mosterrors. Chanceclassaccuracy14.29%. Withheldfont result can be easier thanordinarytest; not evidence of general realaccuracy. Syntheticgenerator constrains widths/layout bycount, a possible shortcut to investigate with independent layouts/generators. Rotationclipping caughtbeforetraining; glyphs scaled tofit, failedlocalattemptretained. All countlabels equal generatedstringlength; seedpartitionsdisjoint; checkpointselection validated; probabilitynormalization passes.

Frozeninference on212 existingfit/reference images across8sources. CV glyphproposalcount also recorded as diagnostic; no CV or labels fitmodel. Modalpredictions: CryptBB5,City5,Courier6,DarkBay6,Outlined6,Roncuevas7,Pymlex4,Sloth6(tie6/7 resolvedlower). These are predictions NOT verifiedfamilylengths. Assistant inspected32examples: City/Pymlex plausible, Courier/Roncuevas oftenovercount, outlined andSloth errors too. Syntheticconfidence can behighonwrongrealcounts. No realaccuracymeasured, no runtimegenerationintegration or automaticcountpromotion.

Model and per-imageprobabilities saved; no priorcheckpoints modified. Nextvalidation should target densecurves/outlines/noise and count-preserving layout shifts before relying on familylengthdistribution. Count range2-8 cannot detect unsupportedlengths; no claim it can. Priorrefinement remainsoptionaloutsidepipeline, automationpaused.

No newpaperadopted: original smallCNN classifier and syntheticaugmentation with existingfont/descriptor utilities. Do not cite DeepFont as a character-count implementation or claim a publishedmethod reproduction.
