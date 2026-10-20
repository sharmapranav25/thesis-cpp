# Phase3 per-parameter audit01

User requested which noncolor adjustments actually help across the bank.320renders:8sources x4exposedreference samples x10conditions (baseline,8singlecontrols,allnoncolor). Tested existingselectedvalues only, not an exhaustiveparametersearch. No training or runtimepromotion. Allbaseline images reproduced exactly; allcolorlocks passed; no renderingfailures. CV freshlymeasured; cached image-matched frozenfeature predictions and priorfrozenfontselection retained.

## Method and limits

Compare baseline against each control alone: height,tilt,spacing,linecount,linewidth,dotcount,curvecount,curveamplitude. Diagnostic distance uses grayscale lowlevel moments/multiscalegradients/projections from fixed_features. It is related to the priorPhase2 metric, NOT independentvalidation or accuracy. Report both familymean and matchedreference distances. Fourexposeddevelopment samples are too few for generalization. Assistant inspected allcontrols for6nonzero families; DarkBay/outlined pixel-identical acrosscontrols because allparameterszero. SharedRNG means count changes may also alter sampledplacement/jitter; attribution is control-level, not perfectlyisolated pixel geometry. No family-specific automaticrule adopted.

## Findings

| Family | Useful candidate | Evidence and reservation |
|---|---|---|
| CryptBB | Linewidth/count, weak candidates | Distances improve7.9%/5.1%; background still lacks structuredgrid. Height score improves but largerletters not clearlyjustified visually. No strong visualwinner. |
| City | Linewidth strongest visualcandidate; linecount secondary | Widerstrokes approach broadreference lines; linewidth distance17.3%better, linecount13.3%. Height23.9%better numerically but not enough todeclare visualwin. |
| Courier | Curvecount, tentative |5.4%family/5.0%paired distanceimprovement; extra curvedinterference directionallymatchesreference, actualgeometrystillpoor. Linecount scoreimproves butstraightlines notclear visualfit. |
| DarkBay | None tested at nonzero | Existingrecipeallzero; no evidence thesecontrols cannothelp, simply nottested. |
| Outlined | None tested at nonzero | Existingrecipeallzero; hollowfontfailureunresolved. |
| Roncuevas | No convincingcandidate | Largestindividualdistancechange only1.5%; incorrectcount/style dominate. |
| Pymlex | Letterheight, tentative |37.4%familydistance and7.7%paired improvement; letters lessundersized inexamples, wrongpixelfont/countremain. Spacing worsens26.5%/20.8%. |
| Sloth | No acceptedcandidate | Height/width give numericalscores butwronghollowfont/extralinesremain. Fullcombination worsensdistance168.8%; do notcall isolatedscoregain success. |

Acrosscurrentvalues, spacing is mostlynegative, curveamplitude negligible, tilt mostlynoop (renderer variant0 disablesrotation forsomefamilies), dotcount mixed/weak. Rounding canmake nonzeroparameters noop. Colorremainsdisabled. No singlecontrolshouldbe globallyenabled basedonthisaudit. Candidatecontrols need broaderdevelopment andnew-family confirmation beforeautomaticselection. Usercurrentsettings unchanged.

No newpaperadopted. This is originallocal ablation; the lowleveldescriptor is reusedimplementation, not a newpublishedbenchmark.
