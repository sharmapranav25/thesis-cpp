# Character count and instance comparison — results

Synthetic metrics only. Real outputs are unlabelled predictions, not accuracy.

| Model | Seed | Selected step | Same exact | Font exact | Layout exact |
|---|---:|---:|---:|---:|---:|
| count_cnn / count_head | 731 | 1100 | 66.14% | 63.64% | 58.41% |
| count_cnn / count_head | 732 | 1100 | 66.14% | 62.05% | 56.82% |
| count_cnn / count_head | 733 | 1000 | 59.09% | 60.45% | 60.00% |
| instance_transformer / count_head | 731 | 1100 | 62.73% | 59.55% | 58.41% |
| instance_transformer / instances | 731 | 1100 | 13.64% | 15.45% | 16.14% |
| instance_transformer / count_head | 732 | 1200 | 64.32% | 61.59% | 55.91% |
| instance_transformer / instances | 732 | 1200 | 26.14% | 24.32% | 24.09% |
| instance_transformer / count_head | 733 | 1200 | 60.45% | 64.32% | 58.41% |
| instance_transformer / instances | 733 | 1200 | 17.73% | 18.86% | 20.23% |

## Layout shortcut audit

A metadata-only ExtraTrees baseline uses requested occupied span and average requested glyph width, not image pixels. This is a diagnostic for residual selection bias, not a proposed real-image estimator. Fit on synthetic training; no choices based on locked tests.
- test: metadata-only exact 0.175; uniform chance 0.143; span/count correlation -0.050.
- font_test: metadata-only exact 0.207; uniform chance 0.143; span/count correlation -0.025.
- layout_test: metadata-only exact 0.196; uniform chance 0.143; span/count correlation 0.007.

## Real style-fit predictions

Counts below are ensemble modal predictions. They are not reference labels. Disagreement and histograms are retained rather than forcing every image to one family length.

### cryptbb

- count_cnn: {8: 36, 10: 26, 9: 1, 6: 1}; three-seed exact agreement 4.7%.
- instance_transformer: {8: 37, 5: 5, 9: 5, 7: 10, 10: 4, 6: 3}; three-seed exact agreement 23.4%.
- Instance-branch median histogram: {12: 37, 9: 3, 11: 15, 10: 7, 8: 2}

### city

- count_cnn: {5: 41, 4: 23}; three-seed exact agreement 60.9%.
- instance_transformer: {5: 14, 4: 50}; three-seed exact agreement 71.9%.
- Instance-branch median histogram: {3: 21, 2: 33, 1: 8, 4: 2}

### courier

- count_cnn: {10: 43, 8: 7, 9: 3, 6: 1, 7: 10}; three-seed exact agreement 32.8%.
- instance_transformer: {10: 35, 7: 16, 9: 5, 6: 2, 5: 1, 8: 5}; three-seed exact agreement 28.1%.
- Instance-branch median histogram: {11: 23, 12: 28, 10: 9, 3: 1, 9: 2, 5: 1}

### darkbay

- count_cnn: {10: 63, 9: 1}; three-seed exact agreement 92.2%.
- instance_transformer: {10: 61, 9: 3}; three-seed exact agreement 79.7%.
- Instance-branch median histogram: {12: 64}

### outlined_text_captcha

- count_cnn: {6: 5, 4: 8, 5: 15, 2: 34, 1: 1, 7: 1}; three-seed exact agreement 1.6%.
- instance_transformer: {4: 31, 3: 18, 2: 14, 5: 1}; three-seed exact agreement 20.3%.
- Instance-branch median histogram: {7: 2, 0: 29, 9: 6, 1: 10, 3: 4, 8: 1, 2: 1, 4: 6, 5: 3, 6: 2}

## Decision

No automatic pipeline promotion. Synthetic scores and real seed agreement do not establish real accuracy. Review visible instance locations, short-length errors, and independent count-only evaluation before changing generation lengths. The existing2–8 policy remains until that evidence is available.