#!/usr/bin/env bash
# Search-R1 SFT v1 MiSS adapter training on rwkv7-g1k-2.9b-20260930-ctx25600:
# data/sft/searchr1-g1k-v1-pack-20261022/train.jsonl (821 rows, Qwen3.7-Flash
# teacher, G1K tool format with thinking, longest row 3649 tokens).
# Copied from RWKV-Agent bench/distill/tools/train-2-2.sh; same layout on the
# Pro6000 server (/mnt/nvme0n1/no22, rwkv_lighting_cuda v1.8.6).
#
#   GPUS="0" SMOKE=1 ./train-miss-sr1-v1.sh   one optimizer step on the 8 longest
#                                             rows for every run at once,
#                                             foreground: the peak memory of
#                                             the parallel sweep shows up here
#   GPUS="0" ./train-miss-sr1-v1.sh           start the lr sweep (nohup), all
#                                             three runs side by side on GPU0
#   GPUS="0" SERIAL=1 ./train-miss-sr1-v1.sh  same, one run after another
#                                             (SERIAL=1 SMOKE=1 checks one run)
#   CHUNK=1024 ...                            smaller state-passing chunk if a
#                                             single run still runs out of memory
#   GPUS="0" ONLY="g1k2.9b-r16-lr5e-5-e2" ... start a single run
#
# GPUS has no default: check nvidia-smi for a free card first. Runs are dealt
# to the listed GPUs round robin, so several runs can share one card: a 2.9B
# run at ctx 4096 is far below the Pro 6000's 96 GB, and rwkv_miss_tune feeds
# one sample at a time, so a single run leaves the card underused anyway.
#
# Run it from the rwkv_lighting_cuda directory (rwkv_miss_tune and the vocab
# live there; the model is found there or one level up). Upload
# data/sft/searchr1-g1k-v1-pack-20261022/train.jsonl as segments-sr1-v1.jsonl
# next to the script:
#   $SFT_HOST:  ~/no22/rwkv_lighting_cuda/{train-miss-sr1-v1.sh,segments-sr1-v1.jsonl}
#   Pro6000:  /mnt/nvme0n1/no22/, linked into rwkv_lighting_cuda/ like train-2-2.sh
# rwkv_miss_tune must support the segments loss mask (Alic-Li/rwkv_lightning_cuda
# main from 47df608, or the v1.8.6 release); the v1.8.0 release would train on
# the whole prompt. Both servers ran segments-2-x.jsonl with their binaries.
#
# Why these settings: 821 rows / batch 8 = 103 updates per epoch, 2 epochs.
# v2.2 used lr 2e-5 on 7.2B with 5953 rows; with 7x fewer rows and a smaller
# model we sweep 2e-5 / 5e-5 / 1e-4 and pick by the eval_base.py rerun.
# ctx 4096 covers every row, so one chunk per row.
#
# A finished run (adapter-final.pth present) is skipped, so rerunning the
# script after an interruption only starts what is missing.
set -euo pipefail
trap 'echo "train-miss-sr1-v1.sh: line $LINENO failed (exit $?): $BASH_COMMAND" >&2' ERR
cd "$(dirname "$0")"

MODEL_FILE=rwkv7-g1k-2.9b-20260930-ctx25600.pth
# $SFT_HOST keeps the model next to rwkv_miss_tune, the Pro6000 server one level up.
if [[ -z ${MODEL:-} ]]; then
  if [[ -e $MODEL_FILE ]]; then MODEL=$MODEL_FILE; else MODEL=../$MODEL_FILE; fi
fi
# Seen on $SFT_HOST 2026-10-22; set MODEL_SHA256_PREFIX= (empty) to skip the check.
MODEL_SHA256_PREFIX=${MODEL_SHA256_PREFIX-d8f8ecd4}
DATA=${DATA:-segments-sr1-v1.jsonl}
DATA_SHA256=6857eb02303034bf9b92c50d4e76f03aff8180874923d9d762f938ff52000975
VOCAB=${VOCAB:-rwkv_vocab_v20230424.txt}
OUT=${OUT:-train-miss-sr1-v1}
BASE_OUT=$OUT
CHUNK=${CHUNK:-4096}
SMOKE=${SMOKE:-0}
SERIAL=${SERIAL:-0}
ONLY=${ONLY:-}
if [[ -z ${GPUS:-} ]]; then
  echo 'set GPUS, e.g. GPUS="0 1 2" (check nvidia-smi for the free cards)' >&2
  exit 1
fi
read -r -a GPU_LIST <<<"$GPUS"

# 103 updates per epoch; a checkpoint every 52 updates is about one per half epoch.
# --wkv_tape: shared WKV replay tape, extra WKV recompute for less memory
# (train-2-2.sh uses it too; without it three runs on one card ran out of
# memory in "tune WKV exact tape" on $SFT_HOST).
COMMON=(--model "$MODEL" --vocab "$VOCAB"
  --targets all --ctx 4096 --chunk "$CHUNK" --wkv_tape
  --batch-size 8 --warmup-steps 10 --save-every 52)

# name rank alpha lr lr-final epochs; lr-final is always lr/10
RUNS=(
  "g1k2.9b-r16-lr2e-5-e2  16 16 0.00002 0.000002 2"
  "g1k2.9b-r16-lr5e-5-e2  16 16 0.00005 0.000005 2"
  "g1k2.9b-r16-lr1e-4-e2  16 16 0.0001  0.00001  2"
)
if [[ -n $ONLY ]]; then
  picked=()
  for run in "${RUNS[@]}"; do
    read -r name _ <<<"$run"
    if [[ " $ONLY " == *" $name "* ]]; then picked+=("$run"); fi
  done
  RUNS=("${picked[@]}")
fi
if (( ${#RUNS[@]} == 0 )); then
  echo "ONLY=$ONLY matches no run" >&2
  exit 1
fi

for f in ./rwkv_miss_tune "$MODEL" "$DATA" "$VOCAB"; do
  [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }
done
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
if [[ $actual != "$DATA_SHA256" ]]; then
  echo "$DATA sha256 $actual, want $DATA_SHA256" >&2
  exit 1
fi
# The adapter records the base model fingerprint; a different file of the
# same name would give adapters the inference server refuses.
if [[ ! -e $OUT/.model-ok ]]; then
  echo "checking $MODEL sha256..."
  model_sha=$(sha256sum "$MODEL" | cut -d' ' -f1)
  if [[ -n $MODEL_SHA256_PREFIX && $model_sha != "$MODEL_SHA256_PREFIX"* ]]; then
    echo "$MODEL sha256 $model_sha, want $MODEL_SHA256_PREFIX..." >&2
    exit 1
  fi
  echo "$MODEL sha256 $model_sha"
  mkdir -p "$OUT"
  echo "$model_sha" >"$OUT/.model-ok"
fi

run_one() {
  local gpu=$1 data=$2 name=$3 rank=$4 alpha=$5 lr=$6 lr_final=$7 epochs=$8 extra=("${@:9}")
  local dir=$OUT/$name log=$OUT/logs/$name
  if [[ -e $dir/adapter-final.pth ]]; then
    echo "$name: done, skipping"
    return 0
  fi
  if [[ -e $dir ]]; then
    echo "$name: $dir exists without adapter-final.pth; move it away first" >&2
    return 1
  fi
  mkdir -p "$OUT/logs"
  local cmd=(./rwkv_miss_tune "${COMMON[@]}" --data "$data" --output "$dir"
    --rank "$rank" --alpha "$alpha" --lr "$lr" --lr-final "$lr_final"
    --epochs "$epochs" "${extra[@]}")
  {
    echo "date: $(date -Is)"
    echo "gpu: $gpu"
    echo "model: $MODEL sha256 $(cat "$OUT/.model-ok" 2>/dev/null || echo '?')"
    echo "data: $data"
    echo "cmd: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
  } >"$log.run.txt"
  echo "$name: start on GPU$gpu $(date -Is) -> $log.log"
  local status=0
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >"$log.log" 2>&1 || status=$?
  echo "$name: exit $status $(date -Is)"
  return "$status"
}

if [[ $SMOKE == 1 ]]; then
  # One step on the longest rows, so the peak memory of the real run shows
  # up here rather than an hour in.
  OUT=$OUT-smoke
  rm -rf "$OUT"
  mkdir -p "$OUT/logs"
  cp "$BASE_OUT/.model-ok" "$OUT/"
  smoke_data=$OUT/longest-8.jsonl
  # awk 'NR<=8', not head: head closes the pipe early, sort dies of SIGPIPE
  # and pipefail turns that into a silent exit 141.
  awk '{print length($0) "\t" NR}' "$DATA" | sort -rn | awk 'NR<=8' | cut -f2 |
    sort -n | awk 'NR==FNR{want[$1]=1; next} want[FNR]' - "$DATA" >"$smoke_data"
  # SERIAL=1 checks one run at a time (the memory of a single run); otherwise
  # all runs start together, as the real parallel sweep would.
  i=0
  pids=()
  fail=0
  for run in "${RUNS[@]}"; do
    read -r name rank alpha lr lr_final epochs <<<"$run"
    if [[ $SERIAL == 1 ]]; then
      run_one "${GPU_LIST[0]}" "$smoke_data" "$name" "$rank" "$alpha" "$lr" "$lr_final" 1 --max-steps 1 || fail=1
    else
      run_one "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" "$smoke_data" "$name" "$rank" "$alpha" "$lr" "$lr_final" 1 --max-steps 1 &
      pids+=($!)
    fi
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  for run in "${RUNS[@]}"; do
    read -r name _ <<<"$run"
    echo "== $name"
    tail -n 5 "$OUT/logs/$name.log"
  done
  if (( fail )); then
    echo "smoke failed; see $OUT/logs. On out of memory: SERIAL=1, then CHUNK=1024" >&2
    exit 1
  fi
  echo "smoke ok; start the real runs with the same GPUS"
  exit 0
fi

mkdir -p "$OUT/logs"
if [[ -n ${RUN_CHILD:-} ]]; then
  read -r gpu run <<<"$RUN_CHILD"
  read -r name rank alpha lr lr_final epochs <<<"$run"
  run_one "$gpu" "$DATA" "$name" "$rank" "$alpha" "$lr" "$lr_final" "$epochs"
  exit $?
fi
if [[ -n ${SERIAL_CHILD:-} ]]; then
  # One run after another on the first GPU; a failed run does not stop the rest.
  for run in "${RUNS[@]}"; do
    read -r name rank alpha lr lr_final epochs <<<"$run"
    run_one "${GPU_LIST[0]}" "$DATA" "$name" "$rank" "$alpha" "$lr" "$lr_final" "$epochs" || true
  done
  exit 0
fi
if [[ $SERIAL == 1 ]]; then
  SERIAL_CHILD=1 nohup "$0" >"$OUT/logs/serial.nohup.log" 2>&1 &
  echo "${#RUNS[@]} runs one after another on GPU${GPU_LIST[0]}, pid $! -> $OUT/logs/serial.nohup.log"
  echo "follow: tail -f $OUT/logs/*.log"
  exit 0
fi
for i in "${!RUNS[@]}"; do
  read -r name _ <<<"${RUNS[$i]}"
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  RUN_CHILD="$gpu ${RUNS[$i]}" nohup "$0" >"$OUT/logs/$name.nohup.log" 2>&1 &
  echo "$name: GPU$gpu pid $! -> $OUT/logs/$name.log"
done
echo "follow: tail -f $OUT/logs/*.log"
