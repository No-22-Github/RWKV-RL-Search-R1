#!/usr/bin/env bash
# 2026-10-27 链路评测：基础模型 → SFT → v1-s100 → v2-s200，按榜单协议测七集抽样（s300 已有结果）
# 用来回答「SFT / 每一轮 RL 各贡献了多少」。服务器上跑，推理 8100 / 检索 8002。
# 用法：bash scripts/eval_lineage_1027.sh 2>&1 | tee outputs/eval-lineage-1027.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
API=http://127.0.0.1:8100

register() {  # register <adapter_id> <服务器本地 pth 路径>
  curl -sS -X POST "$API/v1/adapters" -H 'Content-Type: application/json' \
    --data "{\"adapter_id\":\"$1\",\"path\":\"$(realpath "$2")\"}"; echo
}
register sr1v1-r16-lr5e-5-e2 adapters/sr1v1-r16-lr5e-5-e2.pth
register rl-sr1-v1-s100 adapters/rl-sr1-v1-s100.pth
register rl-sr1-v2-s200 rl-out/server/f2fd88b6304bab67b2ae02b624d2295c_1-step-200/adapter.pth

COMMON="--data data/search_r1/test_upstream7_s300.parquet --format g1k --think full --max_turns 4 --final_answer_turn
  --max_obs_tokens 500 --greedy --max_turn_tokens 2048 --topk 3 --api $API --retriever http://127.0.0.1:8002
  --batch_size 16 --workers 64"
for m in base sr1v1-r16-lr5e-5-e2 rl-sr1-v1-s100 rl-sr1-v2-s200; do
  out=outputs/upstream-eval/lineage-$m-greedy-t3
  adapter=$([ "$m" = base ] || echo "--adapter_id $m")
  echo "[$(date '+%H:%M:%S')] start $m"
  $PY scripts/eval_base.py $COMMON $adapter --out "$out" > "$out.log" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] done  $m" || echo "[$(date '+%H:%M:%S')] FAIL  $m (see $out.log)"
done
echo "[$(date '+%H:%M:%S')] lineage finished"
