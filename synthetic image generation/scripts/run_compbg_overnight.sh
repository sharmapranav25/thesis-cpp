#!/usr/bin/env bash
# Mirror of model 1/run_v2_overnight.sh but pointing at *_compbg configs.
# Trains PARSeq + detector + 3 char-CNN scales, then evals on real CryptBB.
#
# Run from this project root:
#   bash scripts/run_compbg_overnight.sh

set -e
MODEL1="/e/pranav/model 1"
PY="$MODEL1/.venv/Scripts/python.exe"
LOG_DIR="$MODEL1/checkpoints"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

cd "$MODEL1"

echo "[$(ts)] === COMPBG OVERNIGHT PIPELINE BEGIN ==="

echo "[$(ts)] [1/6] Training PARSeq on compbg synth (~30 min)..."
"$PY" -m src.parseq.train --config configs/parseq_compbg.yaml \
  > "$LOG_DIR/parseq_compbg_train.log" 2>&1
echo "[$(ts)]    PARSeq done."

echo "[$(ts)] [2/6] Training detector on compbg synth (~45 min)..."
"$PY" -m src.detector.train --config configs/detector_compbg.yaml \
  > "$LOG_DIR/detector_compbg_train.log" 2>&1
echo "[$(ts)]    detector done."

echo "[$(ts)] [3/6] Training char-CNN 64x64 on compbg synth (~45 min)..."
"$PY" -m src.charcls.train --config configs/charcls_64_compbg.yaml \
  > "$LOG_DIR/charcls_64_compbg_train.log" 2>&1
echo "[$(ts)]    char-CNN 64 done."

echo "[$(ts)] [4/6] Training char-CNN 96x96 on compbg synth (~50 min)..."
"$PY" -m src.charcls.train --config configs/charcls_96_compbg.yaml \
  > "$LOG_DIR/charcls_96_compbg_train.log" 2>&1
echo "[$(ts)]    char-CNN 96 done."

echo "[$(ts)] [5/6] Training char-CNN 128x128 on compbg synth (~80 min)..."
"$PY" -m src.charcls.train --config configs/charcls_128_compbg.yaml \
  > "$LOG_DIR/charcls_128_compbg_train.log" 2>&1
echo "[$(ts)]    char-CNN 128 done."

echo "[$(ts)] [6/6] Running compbg ensemble eval on 651 real captchas..."
BEST_PARSEQ=$(ls checkpoints/parseq_compbg/parseq-epoch=*-val_accuracy=*.ckpt 2>/dev/null \
              | sort -t= -k3 -n -r | head -n1)
if [ -z "$BEST_PARSEQ" ]; then
    echo "[$(ts)]    ERROR: no PARSeq compbg checkpoint found, skipping eval."
    exit 1
fi
echo "[$(ts)]    using PARSeq ckpt: $BEST_PARSEQ"

"$PY" -m src.eval.case_decode \
    --ckpt "$BEST_PARSEQ" \
    --src data/eval --labels data/eval/labels.csv \
    --train-labels data/cryptbb_synth_compbg/labels.csv \
    --detector-ckpt checkpoints/detector_compbg/best.pt \
    --charcls-ckpt "checkpoints/charcls_64_compbg/best.pt,checkpoints/charcls_96_compbg/best.pt,checkpoints/charcls_128_compbg/best.pt" \
    > "$LOG_DIR/compbg_ensemble_eval.log" 2>&1
echo "[$(ts)]    eval done. See $LOG_DIR/compbg_ensemble_eval.log"

echo "[$(ts)] === COMPBG OVERNIGHT PIPELINE COMPLETE ==="
echo ""
echo "Summary lines (compare to v2's 49.16% word / 86.86% char):"
grep -E "ensemble a=|=== summary ===|word%|^  [A-G]\." "$LOG_DIR/compbg_ensemble_eval.log" | tail -20
