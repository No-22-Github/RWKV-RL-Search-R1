#!/usr/bin/env bash
# v4 无人值守链：e5 Flat 环境下重筛题池（提前停止）→ 校验建池 → 自动开训
#
# 前提（睡前手动起好）：
#   8848 tune server 已用新二进制 --resume 89e5b06c..._1-step-300 启动
#   8002 e5 Flat 服务（torch 后端）在线
#
# 用法（服务器，screen 里）：
#   bash chain_v4.sh 2>&1 | tee outputs/v4-chain.log
#   只筛不训：TRAIN=0 bash chain_v4.sh ...
#
# 可调环境变量：TARGET_MIXED（默认 1500）POOL_TARGET（默认 800）STEPS（默认 200）LR（默认 2e-6）
# 早上看这三行即可：最终的 "mixed=... / 已测=..."、"题池行数=..."、"开训: ..."

set -u
[[ "${BASH_SOURCE[0]}" == */* ]] && cd "$(dirname "${BASH_SOURCE[0]}")"

SERVER=${SERVER:-http://127.0.0.1:8848}
RETRIEVER=${RETRIEVER:-http://127.0.0.1:8002}
TOPK=${TOPK:-5}
TARGET_MIXED=${TARGET_MIXED:-1500}
POOL_TARGET=${POOL_TARGET:-800}
TRAIN=${TRAIN:-1}
LR=${LR:-2e-6}
STEPS=${STEPS:-200}
FILTER_OUT=outputs/filter/s300-e5
PY=.venv/bin/python          # 绕开 uv run（它会按 lockfile 把 torch 回滚成 +cpu）

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

count_mixed() {
  "$PY" - <<'PYEOF'
import json, os
p = "outputs/filter/s300-e5/questions.jsonl"
seen = {}
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        seen[(r["qid"], r.get("source"))] = r["correct"]
mixed = sum(1 for v in seen.values() if 0 < v < 8)
print(mixed, len(seen))
PYEOF
}

# ---------- 0) 前置检查 ----------
curl -s -m 8 "$SERVER/health" | grep -q policy_version || { say "FATAL: tune server $SERVER 不在，先起服务端"; exit 1; }
curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1}' >/dev/null || { say "FATAL: 检索 $RETRIEVER 不在"; exit 1; }
STEP=$(curl -s "$SERVER/health" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['step'])")
say "前置检查通过：服务端 step=$STEP，检索 $RETRIEVER topk=$TOPK，目标 mixed=$TARGET_MIXED"

# ---------- 0.5) 内存采集（后台，灌 mem-watch.log） ----------
if ! pgrep -f "mem-watch-loop" >/dev/null; then
  cat > mem-watch-loop.sh <<'EOF'
while sleep 5; do
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  rss=$(ps -o rss= -p "$(pgrep -d, -f rwkv_miss_tune_server)" 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/1024}')
  echo "$(date +%H:%M:%S) gpu=${gpu}MiB server_rss=${rss}MB"
done
EOF
  nohup bash mem-watch-loop.sh >> mem-watch.log 2>&1 &
  say "内存采集已挂后台：mem-watch.log（每 5 秒一行，含显存与训练服务端 RSS）"
fi

# ---------- 1) 分批筛题，凑够即停 ----------
for N in 1600 3200 4800 6400; do
  read -r MIXED SEEN <<< "$(count_mixed)"
  say "当前 mixed=$MIXED / 已测 $SEEN（目标 $TARGET_MIXED）"
  if [ "$MIXED" -ge "$TARGET_MIXED" ]; then
    say "达标，不再续筛"; break
  fi
  say "开始/续筛 n=$N ..."
  ok=0
  for attempt in 1 2 3; do
    if "$PY" scripts/filter_questions.py --server "$SERVER" --retriever "$RETRIEVER" \
         --topk "$TOPK" --group 8 --batch_questions 32 \
         --data data/search_r1/train.parquet --out "$FILTER_OUT" \
         --n "$N" --seed 888 --max_turns 4 --max_turn_tokens 1024 --temperature 1.0; then
      ok=1; break
    fi
    say "筛题异常（attempt $attempt/3），60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { say "FATAL: 筛题连续 3 次失败"; exit 1; }
  read -r MIXED SEEN <<< "$(count_mixed)"
  say "n=$N 段完成：mixed=$MIXED / 已测 $SEEN"
done

# ---------- 2) 题池校验 ----------
POOL=$FILTER_OUT/filtered.parquet
ROWS=$("$PY" -c "
import pandas as pd
df = pd.read_parquet('$POOL')
assert df.duplicated(['id','data_source']).sum() == 0, '池内 (id,data_source) 重复'
print(len(df))
") || { say "FATAL: 题池校验失败（$POOL）"; exit 1; }
say "题池 $POOL 行数=$ROWS（下限 $POOL_TARGET）"
if [ "$ROWS" -lt "$POOL_TARGET" ]; then
  say "FATAL: 题池不足，不开训。可改 TARGET_MIXED/续筛后再跑"; exit 1
fi

# ---------- 3) 自动开训 ----------
if [ "$TRAIN" = "1" ]; then
  say "开训：$ROWS 题 | lr=$LR | 目标 $STEPS 步 | out=outputs/rl/sr1-v4 | 检索 topk=$TOPK"
  "$PY" scripts/rl_grpo.py --server "$SERVER" \
    --data "$POOL" --out outputs/rl/sr1-v4 --steps "$STEPS" --lr "$LR" \
    --f1_weight 0.3 --topk "$TOPK" --retriever "$RETRIEVER" \
    2>&1 | tee outputs/rl-sr1-v4.log
  say "训练进程结束/中断，日志 outputs/rl-sr1-v4.log；指标 outputs/rl/sr1-v4/metrics.jsonl"
else
  say "TRAIN=0，只筛不训。题池就绪：$POOL"
fi
