#!/usr/bin/env bash
# Overnight pipeline: train all components on v2 synth (MyBB fonts only,
# native 200×60, no supersample, no mixed-scale). Total wall ~4 hours.
#
# Usage: bash run_v2_overnight.sh

set -e
PY=./.venv/Scripts/python.exe
LOG_DIR=checkpoints
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] === v2 OVERNIGHT PIPELINE BEGIN ==="

echo "[$(ts)] [1/6] Training PARSeq on v2 synth (~30 min)..."
$PY -m src.parseq.train --config configs/parseq_v2.yaml > "$LOG_DIR/parseq_v2_train.log" 2>&1
echo "[$(ts)]    PARSeq done."

echo "[$(ts)] [2/6] Training detector on v2 synth (~45 min)..."
$PY -m src.detector.train --config configs/detector_v2.yaml > "$LOG_DIR/detector_v2_train.log" 2>&1
echo "[$(ts)]    detector done."

echo "[$(ts)] [3/6] Training char-CNN 64x64 on v2 synth (~45 min)..."
$PY -m src.charcls.train --config configs/charcls_64_v2.yaml > "$LOG_DIR/charcls_64_v2_train.log" 2>&1
echo "[$(ts)]    char-CNN 64 done."

echo "[$(ts)] [4/6] Training char-CNN 96x96 on v2 synth (~50 min)..."
$PY -m src.charcls.train --config configs/charcls_96_v2.yaml > "$LOG_DIR/charcls_96_v2_train.log" 2>&1
echo "[$(ts)]    char-CNN 96 done."

echo "[$(ts)] [5/6] Training char-CNN 128x128 on v2 synth (~80 min)..."
$PY -m src.charcls.train --config configs/charcls_128_v2.yaml > "$LOG_DIR/charcls_128_v2_train.log" 2>&1
echo "[$(ts)]    char-CNN 128 done."

echo "[$(ts)] [6/6] Running v2 multi-scale ensemble eval on 651 real captchas..."
BEST_PARSEQ=$(ls checkpoints/parseq_v2/parseq-epoch=*-val_accuracy=*.ckpt 2>/dev/null \
              | sort -t= -k3 -n -r | head -n1)
if [ -z "$BEST_PARSEQ" ]; then
    echo "[$(ts)]    ERROR: no PARSeq v2 checkpoint found, skipping eval."
    exit 1
fi
echo "[$(ts)]    using PARSeq ckpt: $BEST_PARSEQ"

$PY -m src.eval.case_decode \
    --ckpt "$BEST_PARSEQ" \
    --src data/eval --labels data/eval/labels.csv \
    --train-labels data/synth_mybb_v2/labels.csv \
    --detector-ckpt checkpoints/detector_v2/best.pt \
    --charcls-ckpt "checkpoints/charcls_64_v2/best.pt,checkpoints/charcls_96_v2/best.pt,checkpoints/charcls_128_v2/best.pt" \
    > "$LOG_DIR/v2_ensemble_eval.log" 2>&1
echo "[$(ts)]    eval done. See $LOG_DIR/v2_ensemble_eval.log"

echo "[$(ts)] === v2 OVERNIGHT PIPELINE COMPLETE ==="
echo ""
echo "Summary lines:"
grep -E "ensemble a=|=== summary ===|word%|^  [A-G]\." "$LOG_DIR/v2_ensemble_eval.log" | tail -20
