# Straight revision calibration and real development test01

Accepted synthetic-selection model frozen, SHA b2cf03fa9ab19849279f303ba33115a88cf7d7c0b96527f2c749cf10bf46be99. Synthetic-only calibration yields absent<=.546983 and present>=.785389; otherwise uncertain. In-sample NPV.9023 (6469support), precision.9025 (1169support); not independently validated confidence and not real accuracy. Thresholds saved separately, original models unchanged.

Reran CV from image pixels and frozen straight inference on16 previously exposed sloth reference images/480blocks. Curves,dots and recipe parameters retained from frozen prior inference; botharms use optionalachromaticconstraint. No realtraining or thresholdtuning, no reserveddata touched.

Outcome: all480blocks stillpresent,16/16imagepositive,295straightline operations botharms;32pairedrenders completed. Assistant reviewed4pairs: no visible improvement, remaining lineclutter. Syntheticnegative success did NOT establish realtransfer. Do notpromote this as a stylefix. Rejectedcurve revision remainsunused.

Next work should broaden synthetic texture/antialiasing/scale coverage and examine geometric support for predictedlines. Preserve the distinction between classifier inference and CV confirmation; no family-specific overrides. Currentglobalanypositive aggregation is uncalibrated, though this failure is not merelyaggregation because allblockspositive. New independentfamily evaluation remainsowed for futureacceptedrevision.
