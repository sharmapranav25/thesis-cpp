# synth_v1 — overnight build status

## What's in place

```
src/synth_v1/
  scheme.py        SchemeSpec dataclass + CRYPTBB instance + get(name) lookup
  build_bank.py    detector + parseq → per-char crop bank (alpha-keyed)
  font_bank.py     CharBank: load bank, sample(ch, rng) with jitter
  compose.py       Composer: bank + bg → labeled synth image
  generate.py      CLI: --scheme cryptbb --n N --out DIR  + QA sheet
```

## What ran

```
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v1.build_bank --scheme cryptbb
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v1.generate --scheme cryptbb --n 200 --out data/synth_v1/cryptbb_smoke
```

Bank summary (after 3rd build, with continuous-alpha keying):
- 600 raw → 5 skipped (parseq emitted ≠5 chars or no LaMa pair)
- **2714 alpha-keyed RGBA char crops** across all 62 charset chars
- per-char range: z=1 (filtered out), 0=10, o=16, O=24 → most chars 35–67
- **z** dropped at min_count=5; effective compose charset has **61 chars**

Smoke dataset: 200 imgs in `data/synth_v1/cryptbb_smoke/` + labels.csv + bboxes.jsonl. Render rate ~40/s (PIL-bound, single-process).

QA sheet: `mvp_results/synth_v1_cryptbb_qa.png` — top half synth, bottom half real CryptBB. **Look at this first.**

## What to look at first

1. `mvp_results/synth_v1_cryptbb_qa.png` — the synth-vs-real comparison. Top 4 rows = synth, bottom 4 rows = real.
2. `data/synth_v1/banks/cryptbb/_qa.png` — per-char crop sheet (compact thumbnail per char).
3. `data/synth_v1/banks/cryptbb/_meta.json` — counts, thresholds, ckpt paths used.
4. `data/synth_v1/cryptbb_smoke/0000000.png` etc. — full-res synth samples.
5. Decide whether the residual rectangular halo around some chars is acceptable, or whether to tighten the alpha mask further (notes below).

## What works

- Crayon/colored char style is preserved — synth visually matches real CryptBB at a glance.
- Pseudolabeling pipeline (detector + parseq + per-char conf filter) is clean. 99.2% image keep rate.
- Case-encoded folder names (`L_a` / `U_A` / `0`) work around Windows FS case-insensitivity. CharBank decodes back to single-char keys at sample time.
- Continuous-alpha keying (gradient `(diff − 25) / 65` clipped) + largest-CC band filter cuts most of the bg leakage.

## What didn't work / what I'd revisit

1. **Residual bg around chars.** Even after 3 alpha-keying iterations (hard threshold → tighter threshold + CC → continuous gradient + central-band CC), some crops still carry a faint rectangular bg patch. Where: chars where the original bg has high color variance and triggers above-threshold diffs over a wide bbox area. Cheap next step: replace diff-based mask with an HSV saturation mask — CryptBB chars are saturated, the map bg is muted. ~30 lines, one rebuild.

2. **Case ambiguity in pseudolabels.** PARSeq sometimes predicts upper case for visually small lowercase chars (and vice-versa). The bank inherits this bias — sampling 'A' may give an actual 'a' crop and vice-versa. Filter currently uses confidence > 0.9, which catches the worst cases but not all. If a recognizer trained on this synth misreads case systematically, that's the cause.

3. **Per-char count imbalance.** `z=1, 0=10, o=16, O=24` — these chars are rare in the real captchas (or PARSeq is not confident on them). Compose treats the available charset uniformly, so common chars in synth are not weighted by real-captcha frequency. Probably fine for recognizer training.

4. **Detector hardcoded to 5 chars.** detector_v2 outputs exactly 5 boxes — CryptBB-specific. For any non-5-char scheme (DarkBay=6, City=?), we'd need a detector that handles N. **This is what blocks the "any captcha" claim.** See "Next" below.

5. **Old code is dead but still on disk.** All the difficulty/curriculum machinery (`src/auto_tune/difficulty.py`, the `t` plumbing in `src/generator/parametric.py` and `src/generator/refined.py`, `mvp_plan.md`) is unused by synth_v1 but still present. Did not delete — separate cleanup pass.

## Concrete numbers worth knowing

- detector ckpt: `model 1/checkpoints/detector_v2/best.pt`
- parseq ckpt: `model 1/checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt`
- parseq config: `model 1/configs/parseq_v2.yaml` (img_size [32, 128])
- bank build time: ~5s detector+parseq + ~5s crop loop = ~10s total for 600 imgs
- generate rate: ~40 imgs/s (PIL bound)
- alpha thresholds: ALPHA_LOW=25, ALPHA_HIGH=90, MIN_CC_AREA=12, CROP_PAD=2
- compose target glyph height: 72% of scheme.H
- per-char placement jitter: ±3px x, ±2px y

## Next (when you wake)

In order of expected payoff:

1. **Eyeball the QA sheet** and judge synth-vs-real fidelity. If close enough → train a PARSeq on synth_v1 output, eval on the real CryptBB held-out set, compare to the 49.16%/86.86% baseline. That's the only metric that actually validates this.
2. **If the rectangular halo is too visible**, swap diff-based alpha for HSV saturation-based alpha. ~30 lines in `build_bank.text_mask_from_diff`. Probably 1 hour including rebuild.
3. **Second scheme** — needs (a) a detector that predicts N≠5 boxes for that scheme, and (b) a recognizer for that scheme to pseudolabel. Both are training jobs. Pick a scheme with a useful unlabeled pool (City: 2000 unlabeled, Courier Market: 2510).
4. **DG-Font path** — only if the char-crop bank approach hits a ceiling on a scheme where a critical char appears too rarely (or not at all) in unlabeled data. Currently no scheme is in that state.
