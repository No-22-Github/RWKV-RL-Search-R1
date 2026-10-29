#!/usr/bin/env bash
# v5 无人值守链：训练环境对齐榜单协议，从 s300 重筛题池 → 校验建池 → 自动开训
#
# 与 v4 的差别（全部对齐 SearchAgent Leaderboard / Search-R1 协议）：
#   检索 e5 Flat top-3（v4 是 top-5）；每次检索结果截到 500 token；
#   --max_turns 5 = 4 次可检索 + 1 轮只答（评测侧的 --max_turns 4 --final_answer_turn，逐轮一致）
#
# 前提（手动起好）：
#   8848 tune server 用 --ctx 12288 --resume <v3b step-300 完整检查点> 启动（resume 前看一眼 checkpoint.json 的 null 键）
#   8002 e5 Flat 服务（torch 后端）在线；8100 推理服务训练期间停掉或调小 --decode-batch-size
#
# 用法（服务器，screen 里）：
#   bash chain_v5.sh 2>&1 | tee -a outputs/v5-chain.log
#   只筛不训：TRAIN=0 bash chain_v5.sh ...
#
# 可调环境变量：TARGET_MIXED（默认 1500）POOL_TARGET（默认 800）STEPS（默认 200）LR（默认 2e-6）
#   N_LIST：分批筛题的累计题数（默认 "1600 3200 4800 6400"）
#   FILTER_OUT / RUN_OUT / DATA / RL_LOG：题池目录、训练输出、筛题题源、训练日志（chain_sft2.sh 用它们换起点）

set -u
[[ "${BASH_SOURCE[0]}" == */* ]] && cd "$(dirname "${BASH_SOURCE[0]}")"

SERVER=${SERVER:-http://127.0.0.1:8848}
RETRIEVER=${RETRIEVER:-http://127.0.0.1:8002}
TOPK=${TOPK:-3}
MAX_OBS=${MAX_OBS:-500}
MAX_TURNS=${MAX_TURNS:-5}
TARGET_MIXED=${TARGET_MIXED:-1500}
POOL_TARGET=${POOL_TARGET:-800}
TRAIN=${TRAIN:-1}
LR=${LR:-2e-6}
STEPS=${STEPS:-200}
N_LIST=${N_LIST:-1600 3200 4800 6400}
FILTER_OUT=${FILTER_OUT:-outputs/filter/s300-e5t3}
RUN_OUT=${RUN_OUT:-outputs/rl/sr1-v5}
DATA=${DATA:-data/search_r1/train.parquet}
RL_LOG=${RL_LOG:-outputs/rl-sr1-v5.log}
PY=.venv/bin/python          # 绕开 uv run（它会按 lockfile 把 torch 回滚成 +cpu）
ENV_ARGS="--topk $TOPK --max_obs_tokens $MAX_OBS --max_turns $MAX_TURNS --max_turn_tokens 1024"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

count_mixed() {
  "$PY" - "$FILTER_OUT/questions.jsonl" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
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
read -r STEP CTX <<< "$(curl -s "$SERVER/health" | "$PY" -c "import json,sys; h=json.load(sys.stdin); print(h['step'], h['ctx'])")"
[ "$CTX" -ge 12288 ] || { say "FATAL: 服务端 ctx=$CTX，4 次检索 + 5 轮生成需要 --ctx 12288"; exit 1; }
grep -q "max_obs_tokens" scripts/rl_grpo.py scripts/filter_questions.py || { say "FATAL: 脚本没同步（缺 --max_obs_tokens）"; exit 1; }
grep -q "zip(df.id, df.data_source)" scripts/filter_questions.py || { say "FATAL: filter_questions.py 还是按裸 id 建池的旧版"; exit 1; }
say "前置检查通过：服务端 step=$STEP ctx=$CTX，检索 topk=$TOPK 截 $MAX_OBS，max_turns=$MAX_TURNS，目标 mixed=$TARGET_MIXED"

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
  say "内存采集已挂后台：mem-watch.log"
fi

# ---------- 1) 分批筛题，凑够即停 ----------
for N in $N_LIST; do
  read -r MIXED SEEN <<< "$(count_mixed)"
  say "当前 mixed=$MIXED / 已测 $SEEN（目标 $TARGET_MIXED）"
  if [ "$MIXED" -ge "$TARGET_MIXED" ]; then
    say "达标，不再续筛"; break
  fi
  say "开始/续筛 n=$N ..."
  ok=0
  for attempt in 1 2 3; do
    if "$PY" scripts/filter_questions.py --server "$SERVER" --retriever "$RETRIEVER" $ENV_ARGS \
         --group 8 --batch_questions 32 --data "$DATA" --out "$FILTER_OUT" \
         --n "$N" --seed 888 --temperature 1.0; then
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
ROWS=$("$PY" - "$POOL" "$FILTER_OUT/questions.jsonl" <<'PYEOF'
import json, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
assert df.duplicated(['id', 'data_source']).sum() == 0, '池内 (id,data_source) 重复'
lab = {(r['qid'], r['source']) for r in map(json.loads, filter(str.strip, open(sys.argv[2])))}
miss = sum((i, s) not in lab for i, s in zip(df.id, df.data_source))
assert miss == 0, f'{miss} 道题没有筛题标签（搭车题）'
print(len(df))
PYEOF
) || { say "FATAL: 题池校验失败（$POOL）"; exit 1; }
say "题池 $POOL 行数=$ROWS（下限 $POOL_TARGET）"
if [ "$ROWS" -lt "$POOL_TARGET" ]; then
  say "FATAL: 题池不足，不开训。可改 TARGET_MIXED/续筛后再跑"; exit 1
fi

# ---------- 3) 自动开训 ----------
if [ "$TRAIN" = "1" ]; then
  say "开训：$ROWS 题 | lr=$LR | 目标 $STEPS 步 | out=$RUN_OUT | $ENV_ARGS"
  "$PY" scripts/rl_grpo.py --server "$SERVER" --retriever "$RETRIEVER" $ENV_ARGS \
    --data "$POOL" --out "$RUN_OUT" --steps "$STEPS" --lr "$LR" --f1_weight 0.3 \
    2>&1 | tee -a "$RL_LOG"
  say "训练进程结束/中断，日志 $RL_LOG；指标 $RUN_OUT/metrics.jsonl"
else
  say "TRAIN=0，只筛不训。题池就绪：$POOL"
fi
