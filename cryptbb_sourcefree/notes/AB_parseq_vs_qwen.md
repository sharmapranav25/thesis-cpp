# A/B: PARSeq (from-scratch specialist) vs Qwen2-VL-2B (pretrained VLM + LoRA)

**Question.** On the *identical* source-free synthetic corpus, how far apart are a
from-scratch font-specialist (PARSeq) and a pretrained font-general VLM (Qwen2-VL-2B)?
This isolates the one variable that the whole thesis hinges on: **model class /
training regime**, with the data held constant.

**Hypothesis.** The gap is caused by the *model*, not the training loop. PARSeq
learns the synth's (generic-font) glyph shapes exactly and can't read MyBB's real
font it never saw → near-0% word. Qwen keeps a broad pretrained font prior and only
adapts it → reads the real font despite the synth font mismatch. Prediction:
PARSeq ≈ 0–4% word, Qwen ≈ 20–30% word on the same pool.

---

## 1. The fairness protocol (this is the whole point)

For the comparison to mean anything, **only the model + its loop may vary.**
Everything below is held identical for both arms.

| Held IDENTICAL (controlled) | Free to differ per model (each tuned to its own optimum) |
|---|---|
| Synth corpus (same images, same labels) | Architecture / param count |
| Train / val split (same rows, same seed) | Trained params (all weights vs LoRA adapters) |
| Real eval set (651 CryptBB) | Objective (char-CE vs autoregressive text-gen) |
| Charset, captcha length (5) | lr, batch, #steps/epochs, optimizer schedule |
| Augmentation policy (light photometric — see note) | Input resolution / tokenization |
| Decode + scoring (force-length-5, word + char) | — |
| **Checkpoint selection = best SYNTH-VAL word** (no eval peeking) | — |
| Real eval scored **exactly once**, at the end | — |

**Two rigor rules, non-negotiable:**
1. **Select by synth-val, not real.** `train_scorer.py` already does this. `qwen_finetune.py`
   currently selects by real-word (eval peeking) — **must be patched** to select by a
   synth-val holdout (see §4). Otherwise the A/B is unfair (Qwen would be tuned on the test set).
2. **Score the 651 real once.** Both arms report a single final number from the
   synth-val-selected checkpoint. No per-epoch real tuning.

**Augmentation note.** Heavy geometric aug (model 1's default) stalls training on already-
distorted synth (documented: loss stuck ~3.3). Both arms use the *same light photometric
aug* (`train_scorer.light_aug` ≈ `qwen_finetune.light_aug`) so aug is a controlled constant,
not a confound.

---

## 2. Corpus size — recommendation: **100k**, length-5, one shared pool

**Generate 100,000 labeled length-5 CryptBB samples** with the source-free PHP-GD renderer.

Reasoning:
- **Size to the hungrier model.** PARSeq is trained from scratch (12-layer ViT, ~24M params)
  and is data-hungry — the notes record it converging slowly / needing init below ~100k.
  100k is also model 1's proven size, so PARSeq is not starved.
- **Qwen shares the identical pool** but only samples a fraction of it (it converges in a few
  thousand steps). That's expected and fair: each model trains to its own convergence on the
  *same* data; the pool size just guarantees neither is data-limited.
- **Cheap:** php_pool renders 100k in ~60s on 32 cores.
- **Length-5 only** to match the 5-char eval and the force-length-5 decode (both arms).

Optional (only if you want a data-efficiency curve, not required for the headline A/B):
also run at **10k and 40k** to see how each model scales with corpus size — PARSeq should
be far more size-sensitive than Qwen. Keep the split seed fixed across sizes.

### Generate the pool + materialize ONE shared split

```bash
cd "E:/pranav/cryptbb_sourcefree"
# 100k source-free PHP-GD synth (generic fonts; labels are ours)
python src/php_pool.py --scheme cryptbb --n 100000 --fixed-len 5 \
  --out data/synth_pool/ab_cryptbb_100k
```

Then split **once** into shared train/val files so both trainers consume identical rows
(5% val holdout, seed 0). Materializing the split to files removes any split-mismatch risk:

```bash
python - <<'PY'
import csv, random
from pathlib import Path
root = Path("data/synth_pool/ab_cryptbb_100k")
rows = list(csv.reader(open(root/"labels.csv")))[1:]
random.Random(0).shuffle(rows)
nval = int(len(rows)*0.05)
val, train = rows[:nval], rows[nval:]
for name, r in [("labels_train.csv", train), ("labels_val.csv", val)]:
    with open(root/name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename","label"]); w.writerows(r)
print("train", len(train), "val", len(val))
PY
```

Both arms train on `labels_train.csv`, select on `labels_val.csv`, eval on
`E:/pranav/model 1/data/eval` (651 real). Guardrail: the 600 unlabeled pool and the 651
eval are never training inputs here — this A/B is synth-only fine-tune + one real score.

---

## 3. Arm A — PARSeq (from-scratch specialist)

**Trainer:** `src/train_scorer.py` (already built for exactly this: light aug, force-length
synth-val selection, `refine_iters=0` for decode). **Config:** `configs/parseq_pool.yaml`
— point its `data_dir` at the shared pool.

**Loop spec:**
- Init: **from scratch** (random). This is the honest source-free specialist test.
  *(Variant, diagnostic only: `--mae-init` from a source-free MAE ckpt, or `--init parseq_v2`
  which taints purity via MyBB fonts — run these only as separate labelled rows, not the headline.)*
- Model: downsized PARSeq — embed 384, enc depth 12 / 6 heads, dec depth 1 / 12 heads,
  `perm_num=1`, `refine_iters=0` (decode), `max_label_length=5`, input 32×128.
- Objective: PARSeq permutation-LM char cross-entropy (`model.training_step`).
- Optim: AdamW, **lr 6e-4**, cosine schedule + 5% warmup, batch **128**, grad-clip 5, fp32.
- Length: **~18 epochs** over 95k (≈ 13k steps). Select best synth-val word.
- Selection: best **force-length-5 synth-val word** on `labels_val.csv`.

```bash
# ensure configs/parseq_pool.yaml: data_dir: data/synth_pool/ab_cryptbb_100k
python src/train_scorer.py --config configs/parseq_pool.yaml \
  --epochs 18 --lr 6e-4 --batch 128 --val-frac 0.05 \
  --out data/ckpts/ab_parseq/best.ckpt
# (optional diagnostic: add --eval-real to watch real-651 per epoch; do NOT select on it)
```

Then score the 651 real **once** from `best.ckpt` (force-len-5 word + char).

---

## 4. Arm B — Qwen2-VL-2B (pretrained VLM + LoRA)

**Trainer:** `src/qwen_finetune.py`. **Required patch for fairness:** it currently scores +
selects on the 651 real. For the A/B, change selection to a **synth-val holdout** and score
real once. Minimal change: load `labels_val.csv`, add a `score_synth_val()` (same force-decode
as its real scorer but on synth val), select the adapter by synth-val word, and call the real
scorer a single time after training.

**Loop spec (baseline, the proven config):**
- Base: `Qwen/Qwen2-VL-2B-Instruct`, **frozen**; train **LoRA r16** (α32, dropout .05) on
  `q_proj,k_proj,v_proj,o_proj`. bf16 + gradient checkpointing.
- Objective: autoregressive next-token CE on the answer only (prompt tokens masked to −100).
- Prompt: *"What are the 5 characters in this captcha? Reply with only the 5 characters…"*
- Optim: AdamW, **lr 1e-4**, batch **2 × accum 4** (eff. 8), grad-clip 1.0.
- Input: captcha resized to ~196px long side (limits visual tokens).
- Length: **~3000 steps** (longer than the historical 1500 — the loss was still falling;
  this A/B should let it plateau). Eval synth-val every 250; select best synth-val word.

```bash
# after patching qwen_finetune.py to select on synth-val (see above)
python -u src/qwen_finetune.py --pool data/synth_pool/ab_cryptbb_100k \
  --eval-dir "E:/pranav/model 1/data/eval" --n-len 5 \
  --steps 3000 --eval-every 250 --lr 1e-4 \
  --out data/ckpts/ab_qwen
```

Then score the 651 real **once** from the synth-val-best adapter.

---

## 5. What to record

Fill this per run (headline = from-scratch PARSeq vs LoRA-r16 Qwen, 100k, identical split):

| Arm | Init / trained params | Corpus | Steps seen | Synth-val word | **Real word** | Real char | GPU-hrs |
|---|---|---|---|---|---|---|---|
| PARSeq from-scratch | random / ~24M all | 100k | | | | | |
| Qwen2-VL LoRA r16 | frozen 2B / ~10M adapter | 100k | | | | | |

Expected shape of the result (prior evidence): PARSeq ≈ 0–4% real word, Qwen ≈ 20–30%.
The **size of that gap on identical data is the finding** — it attributes the difference to
the model class (pretrained font prior), not to the data or the loop.

---

## 6. Optional extension arms (run only after the headline A/B)

Each is a *separate labelled row*, not a change to the controlled comparison:

- **Data-efficiency curve:** repeat both arms at 10k / 40k / 100k → PARSeq should be steep,
  Qwen nearly flat (quantifies "the VLM needs less data").
- **Capacity-bumped Qwen:** LoRA r32 + add `mlp` (gate/up/down) and a vision-tower LoRA, more
  steps — tests "more adaptation capacity helps" *without* full fine-tune (which risks erasing
  the general prior — the exact failure mode that gives PARSeq 0%). Keep this modest.
- **Font-match ablation (measures the model-1 'cheat' effect):** regenerate the pool with the
  MyBB font and rerun both arms. Prediction: PARSeq jumps to ~40% (it now sees matched shapes);
  Qwen jumps too but by less (it was already reading the font). The delta quantifies exactly how
  much of model 1's win was the font match vs the model.
```
