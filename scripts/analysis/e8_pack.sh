#!/bin/bash
set -uo pipefail
SEGS=(w024 w043)
SEG=${SEGS[$((FB_TASK_ID-1))]}
CFGS=(ck010k__L04-20 ck040k__L06-22 ck075k__L08-24 ck075k__L04-20)
WORKERS=4
OUT=${OUT_ROOT:-/data/results/E8-pack}/$SEG
mkdir -p $OUT
echo "task=$FB_TASK_ID seg=$SEG host=$(hostname) cores=$(nproc)"
if [ -z "${DRY:-}" ]; then
  source /data/venv/bin/activate
  cd /data/villa/ink-detection
fi
infer_one() {
  local CFG=$1 D=$2
  local CK=${CFG%%__*} LY=${CFG##*__}
  local LS=$((10#${LY:1:2}))
  local LE=$((LS+17)) STEP
  case $CK in
    ck010k) STEP=010000 ;;
    ck040k) STEP=040000 ;;
    ck075k) STEP=075000 ;;
  esac
  local S=$(date +%s.%N)
  if [ -n "${DRY:-}" ]; then
    sleep 1; RC=0
  else
    python -m koine_machines.inference.infer /data/segments/$SEG/surface.zarr \
      /data/models/ink_9um/hybrid_3d2d-seed42/step-$STEP.pth $D/${SEG}__${CFG}.tif \
      --no-compile --num-workers $WORKERS --batch-size 128 \
      --layer-start $LS --layer-end $LE > $D/log_$CFG.txt 2>&1
    RC=$?
  fi
  echo "$CFG $RC $S $(date +%s.%N)" > $D/done_$CFG.txt
}
for K in 1 2 3 4; do
  D=$OUT/K$K
  mkdir -p $D
  rm -f $D/done_*.txt
  nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -l 1 > $D/gpu.csv 2>/dev/null &
  SMI=$!
  T0=$(date +%s.%N)
  for j in $(seq 0 $((K-1))); do
    infer_one ${CFGS[$j]} $D &
  done
  wait $(jobs -p | grep -v "^$SMI$")
  T1=$(date +%s.%N)
  kill $SMI 2>/dev/null
  echo "$K $T0 $T1" > $D/batch.txt
  echo "K=$K concurrent runs: wall $(python3 -c "print(round($T1-$T0,1))") s"
  cat $D/done_*.txt
done
FAIL=$(cat $OUT/K*/done_*.txt | awk '$2!=0' | wc -l)
echo "failed runs: $FAIL"
exit 0  # always let analyze run; failures are reported in its table
