# Existing-corpus simple CNN completed

The user requested a simple CNN on the synthetic data already generated, followed by frozen real inference. This run uses the DarkBay CAPRI_v2 100k corpus, without generating a new count corpus or using the newer35-font smoke images.

Training completed in123.9seconds. Epoch1 was selected by synthetic selection loss. Epoch2 and3 had higher selection loss; the predeclared patience2 stopped training after3epochs. This does not by itself identify whether later deterioration was optimisation instability or overfitting.

## Synthetic results

- Train89,714, selection5,144, locked test5,142. Synthetic text-hash disjoint partitions; original fonts and style profiles shared.
- Locked exact count99.4166%; MAE0.01050.
- Length2:98.59%;3:98.54%;4:99.57%;5:99.62%;6:99.86%;7:99.86%;8:100%.
- The CNN has three convolution blocks and a7-way count head. No transformer, localisation stage, feature pop, real fitting or recognised character identity.

The existing corpus retains count-dependent character fitting and placement. This result is in-distribution synthetic generalisation, not evidence that those shortcuts were removed. Scores must not be treated as a matched architecture comparison against the previous harder synthetic count corpus.

## Entire real DarkBay pool

Frozen inference on799images; sibling eval directory excluded. No real labels were loaded and no checkpoint was selected from real predictions.

| Predicted length | Images |
|---|---:|
|2|0|
|3|0|
|4|3|
|5|18|
|6|357|
|7|6|
|8|415|

These are predictions, not measured real accuracy. Assistant reviewed the fixed first24examples in darkbay_predictions.png: many visibly six-glyph examples receive8, including high-softmax-score predictions. This demonstrates overcounting and makes the score unsuitable as a real-domain confidence guarantee. The preview is not an independent annotation audit of all799images.

## Decision

The requested straightforward baseline is trained and preserved. It learns the existing synthetic task very well but does not yet provide reliable real length inference. Do not automatically change the generator's length distribution to this bimodal6/8histogram. Keep the checkpoint for controlled follow-up; no background-specific rule or forced-six correction was added. The underlying recogniser and generated100k images remain unchanged.

Exact split membership, model, training history, selected checkpoint, probabilities and real prediction gallery are saved beside this report. No real count accuracy is claimed without independent count-only labels.
