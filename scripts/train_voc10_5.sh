#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/kd2"
PYTHON_BIN="${PYTHON:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to the VOC dataset root}"
: "${STEP0_CHECKPOINT:?Set STEP0_CHECKPOINT to the approved future-background0/no-PTC checkpoint}"
: "${CLUE_GPUS:?Set CLUE_GPUS to exactly two available physical GPU IDs, e.g. 4,5}"
MODE="${KD_MODE:-plain}"
RUN_NAME="${RUN_NAME:-voc10_5_${MODE}}"
case "$MODE" in plain|counter|relations|evidence) ;; *) echo "Unsupported KD_MODE: $MODE" >&2; exit 2;; esac
if [[ ! "$CLUE_GPUS" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "CLUE_GPUS must contain exactly two GPU IDs" >&2
  exit 2
fi
if [[ "${CLUE_GPUS%,*}" == "${CLUE_GPUS#*,}" ]]; then
  echo "CLUE_GPUS must contain distinct GPU IDs" >&2
  exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_NAME must use letters, digits, underscores or hyphens" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$CLUE_GPUS"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 USE_LIBUV=0
PREVIOUS="$STEP0_CHECKPOINT"
for STEP in 1 2; do
  "$PYTHON_BIN" -B -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --task 10-5 --step "$STEP" --data_folder "$DATA_ROOT" \
    --list_folder "$REPO_ROOT/kd2/datasets/voc" --work_dir "runs/$RUN_NAME" \
    --prev_checkpoint "$PREVIOUS" --max_iters 8000 --eval_iters 2000 --log_iters 50 \
    --spg 4 --num_workers 4 --crop_size 448 --scales .5 2 --cam_scales 1 .5 1.5 \
    --lr 2e-5 --warmup_iters 2000 --warmup_ratio 1e-6 --wt_decay .01 \
    --w_ptc .2 --w_seg .1 --bkg_thre .5 --high_thre .75 --low_thre .25 \
    --old_cls_threshold 2 --aux_layer -3 --seed 0 \
    --recovery_mode local --recovery_start 2000 \
    --fusion_mode cosine --fusion_margin .1 --fusion_start 2000 \
    --kd_mode "$MODE" --w_kd .1 --kd_temperature 2 --kd_teacher_threshold .75 --kd_start 2000
  PREVIOUS="$REPO_ROOT/kd2/runs/$RUN_NAME/10-5/step$STEP/model_final.pth"
done
