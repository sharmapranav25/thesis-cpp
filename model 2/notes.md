# Model 2 Arc — Working Notes

## M2 (arc) — Static Curriculum, Additive 5/5/15

### Setup
- Data: `data/synth_buckets` (15k easy d=0.2 + 25k medium d=0.5 + 57k hard d=1.0)
- Schedule: 5 ep easy → 5 ep easy+medium (50/50) → 15 ep all-on proportional (15/25/60)
- 25 epochs total, from scratch, **no augmentation**, no length head, no SS
- Config: `configs/parseq_arc_m2.yaml`
- Output: `checkpoints/parseq_arc_m2/`

### Key per-epoch val_accuracy curve (mixed val: 15/25/60)

| Ep | val_acc | Stage |
|---|---|---|
| 0–2 | 0.26 → 0.54 | easy only (random) |
| 3 | **12.54** | easy phase transition |
| 4 | 9.10 | easy (last) — `stage_end_easy_epoch04.ckpt` |
| 5 | 19.84 | easy+medium (1st) |
| 7 | **34.32** | easy+medium peak |
| 9 | 16.26 | easy+medium (last) — `stage_end_easy_medium_epoch09.ckpt` |
| 10 | 10.06 | hard turns on (drop) |
| 11 | 4.90 | hard (low) |
| 12 | 16.68 | hard (climbing) |
| 13 | **42.82** | hard phase transition (best ckpt) |
| 14–16 | 35.92 → 12.08 | volatility |
| 17–20 | 27 → 28 | recovery |
| 21 | **39.06** | second peak |
| 22–24 | (in flight) | |

### Hard-image exposure math
- M2: 15 ep × 0.60 hard weight × 95k samples = 855k hard samples → ~15× per hard image
- M1 v2: 15 ep × 100% hard × 100k samples = 1500k hard samples → ~15× per hard image
- **Per-image hard exposure is comparable**, despite the delayed introduction in M2.

### Comparison vs M1 v2 val curve
- M1 v2 phase-transitioned on hard at epoch 4 (31.40%) from random init.
- M2 phase-transitioned on hard at "hard-epoch 4" (epoch 13, 42.82%) from curriculum-pretrained init.
- **At the same hard-training depth, M2 was ~10pts ahead of M1 v2.** That's the curriculum signal.

### What unblocked vs prior bucket-curriculum failure
The previous `parseq_bucket_curriculum` run (v1) never escaped 3.48% val. Two changes here:
1. **`augment: false`** — the prior run had aug ON (default true). The M2 v1 writeup independently identified augmentation as the silent killer (it took M1 v2 from 39.78% → 6.76% in one epoch when fine-tuning).
2. **Additive schedule** (1/0/0 → 1/1/0 → 0.15/0.25/0.60) instead of blend-and-dilute (1/0/0 → 0.5/0.5/0 → 0.2/0.6/0.2 → 0.1/0.4/0.5 → 0/0.2/0.8 → 0/0/1).

Augmentation is the more likely cause based on diagnostic evidence.

### Hypothesis: figure-ground separation is a distinct inductive bias
Stages 1–2 (easy d=0.2, medium d=0.5) trained character recognition with minimal background noise. Hard d=1.0 introduces heavy random backgrounds + lines + dots.

The post-hard regression (ep 13 → ep 16, 42 → 12) is consistent with the encoder having no figure-ground features and needing to learn them mid-training. The recovery (ep 17 → 21, 27 → 39) is consistent with that bias being learned.

**Implication for Model 3 (adaptive curriculum):** decouple the renderer knobs so background-density and char-distortion can be ramped independently. The current single-`d` knob entangles them.

### Stage-end checkpoints (all preserved by `StageEndCheckpointCallback`)
- `stage_end_easy_epoch04.ckpt` — pure easy specialist
- `stage_end_easy_medium_epoch09.ckpt` — easy+medium specialist
- `stage_end_all_epoch24.ckpt` — final (after hard stage)

Plus best-by-val-accuracy: `parseq-epoch=13-val_accuracy=42.820.ckpt` (current MVP candidate).

### Open questions for real-CryptBB eval
1. Does ep 13 (high val, only ~3 effective hard epochs) beat 39.78%? Or is it riding stage-2 specialty?
2. Does any later stage-3 ckpt (lower val, more hard exposure) beat ep 13 on real eval?
3. Does the curriculum trajectory translate to a smaller val→real gap than M1 v2's 17pt gap?

### Infrastructure added in this session
- `StageEndCheckpointCallback` in `src/parseq/train.py` — saves a tagged ckpt at specified epoch ends, independent of top-k pruning.
- `configs/parseq_arc_m1.yaml` — placeholder shuffled baseline (NOT NEEDED — M1 v2 IS the arc's M1).
- `configs/parseq_arc_m2.yaml` — additive 5/5/15 curriculum config.

### Next steps
1. Wait for ep 24 to finish.
2. Run `infer.py` on all 4 key ckpts (`stage_end_easy`, `stage_end_easy_medium`, `parseq-epoch=13`, `stage_end_all`) against `data/eval/` (651 real CryptBB).
3. Compare each to M1 v2's 39.78% bar.
4. If any beats it: M2 is a curriculum win. If not but ep 13 ≈ 39.78%: tied, curriculum produced parity with less hard exposure.
5. For Model 3: decouple renderer knobs (especially separate `bg-pattern` from char distortion) to enable adaptive per-knob curriculum.
