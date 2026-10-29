#!/usr/bin/env bash
# 2026-10-27 评测链（服务器上跑，推理 8100 / 检索 8002）：
#   1) s110 七集（榜单协议：top-3、4 次检索 + 1 轮只答、观察截 500）
#   2) s100 / s300 内部口径（200 题 × 8，top-5，T0.3 / top_p 0.5，3 次检索 + 提示收尾）
#   3) s100 / s300 训练式收尾（七集，top-3，贪心，3 次检索 + 提示收尾）
# 用法：bash scripts/eval_chain_1027.sh 2>&1 | tee outputs/eval-chain-1027.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--format g1k --think full --max_turns 4 --max_turn_tokens 2048 --api http://127.0.0.1:8100 --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64"
UP7=data/search_r1/test_upstream7_s300.parquet

run() {  # run <out> <args...>
  local out=$1; shift
  echo "[$(date '+%H:%M:%S')] start $out"
  $PY scripts/eval_base.py $COMMON "$@" --out "$out" > "$out.log" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] done  $out" || echo "[$(date '+%H:%M:%S')] FAIL  $out (see $out.log)"
}

mkdir -p outputs/upstream-eval outputs/rl-eval
run outputs/upstream-eval/v4-s110-greedy-t3 --data $UP7 --final_answer_turn --max_obs_tokens 500 --greedy --topk 3 --adapter_id rl-sr1-v4-s110
for m in v4-s100 v3b-s300; do
  run outputs/rl-eval/rl-sr1-$m-e5flat-topk5-pro6000 --data data/search_r1/test_sample.parquet --n 8 \
      --temperature 0.3 --top_p 0.5 --topk 5 --adapter_id rl-sr1-$m
done
for m in v4-s100 v3b-s300; do
  run outputs/upstream-eval/$m-greedy-t3-nofinal --data $UP7 --max_obs_tokens 500 --greedy --topk 3 --adapter_id rl-sr1-$m
done
echo "[$(date '+%H:%M:%S')] chain finished"
