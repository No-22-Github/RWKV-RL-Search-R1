#!/usr/bin/env bash
# SFT v2 → RL 第二段：用 step-100 策略重筛新题扩池，从 step-100 完整检查点续训
#
# 背景（2026-10-28 榜单协议七集）：SFT v2 0.379 → s50 0.380 → s100 0.397（format_ok 0.99、检索 2.12→2.48，
# 多跳集回升 = 行为期开始）。旧池只有 734 题、已跑到第 3 轮、mixed 降到 8.6/16，原池续训会背题。
#
#   1) 停推理服务 8100 腾显存，修 step-100 checkpoint.json 的 null 键（上游 bug）
#   2) 新题源 = train_rl_sft2.parquet 去掉第一轮筛过的全部题（旧池不再进新池）
#   3) 训练服务端 --resume step-100 冷启动（adapter 和 Adam 状态都从检查点恢复）
#   4) chain_v5.sh：s100 策略分批筛到 TARGET_MIXED 题 → 建池 → GRPO 100 步（服务端 step 100→200）
#
# 用法（服务器包根目录，screen 里）：
#   bash chain_sft2_r2.sh 2>&1 | tee -a outputs/sft2-r2-chain.log
# 评测：客户端第 49/99 步 = 服务端 step-150/200 完整检查点，发布目录见 metrics.jsonl 的 publish.directory

set -u
[[ "${BASH_SOURCE[0]}" == */* ]] && cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd)
PY=.venv/bin/python
API=http://127.0.0.1:8100
RETRIEVER=http://127.0.0.1:8002
SERVER=http://127.0.0.1:8848
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
SRV=$ROOT/rwkv_lighting_cuda
RESUME=${RESUME:-rl-out/server/d53cd0acfc854114943d3b2a4e027a68_1-step-100}
OLD_FILTER=outputs/filter/sft2-e5t3
SRC=data/search_r1/train_rl_sft2.parquet
DATA=data/search_r1/train_rl_sft2_r2.parquet
# 第一轮 1600 题出 734 道 mixed（46%）；目标 1000 题约筛 2400 题（约 3.5–4 小时），100 步每题约抽 1.6 次
export TARGET_MIXED=${TARGET_MIXED:-1000}
export POOL_TARGET=${POOL_TARGET:-800}
export N_LIST=${N_LIST:-1600 2400 3200 4000}
export STEPS=${STEPS:-100}

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }

# ---------- 0) 前置检查 ----------
curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1}' >/dev/null || die "检索 $RETRIEVER 不在"
curl -s -m 3 "$SERVER/health" >/dev/null && die "8848 已有服务在跑，先停掉"
[ -e "$RESUME/training/checkpoint.json" ] || die "$RESUME/training/checkpoint.json 不在"
[ -e "$OLD_FILTER/questions.jsonl" ] || die "第一轮筛题记录 $OLD_FILTER/questions.jsonl 不在"
grep -q 'N_LIST' chain_v5.sh || die "chain_v5.sh 是旧版（不认 N_LIST），先同步"
say "前置检查通过"

# ---------- 1) 停推理服务，修检查点 ----------
if curl -s -m 2 "$API/v1/adapters" >/dev/null; then
  say "停推理服务 8100"
  screen -S infer -X quit 2>/dev/null
  pkill -f "rwkv_lighting_cuda --model-path .*--port 8100" 2>/dev/null
  for _ in $(seq 1 30); do curl -s -m 2 "$API/v1/adapters" >/dev/null || break; sleep 2; done
  curl -s -m 2 "$API/v1/adapters" >/dev/null && die "8100 停不掉"
fi
# 长命进程写的第 2 个及以后的检查点带 "planned_steps": null / "seed": null，resume 比对 config 会失败
sed -i '/"planned_steps" : null,/d; /"seed" : null,/d' "$RESUME/training/checkpoint.json"
# "scheduler" : null 是正常配置（step-50 也有），只查 bug 插入的两个键
grep -Eq '"(planned_steps|seed)" : null' "$RESUME/training/checkpoint.json" && die "checkpoint.json 里还有 planned_steps/seed 的 null 键，手动看一下"

# ---------- 2) 新题源：去掉第一轮筛过的题 ----------
"$PY" - "$SRC" "$OLD_FILTER/questions.jsonl" "$DATA" <<'PYEOF' || die "生成新题源失败"
import json, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
used = {(str(r["qid"]), str(r["source"])) for r in map(json.loads, filter(str.strip, open(sys.argv[2])))}
keep = [(str(i), str(s)) not in used for i, s in zip(df.id, df.data_source)]
out = df[keep].reset_index(drop=True)
assert len(df) - len(out) == len(used), f"去重数 {len(df) - len(out)} != 已筛 {len(used)}（id/source 对不上）"
out.to_parquet(sys.argv[3])
print(f"题源 {len(df)} - 已筛 {len(used)} = {len(out)} -> {sys.argv[3]}", file=sys.stderr)
PYEOF

# ---------- 3) 训练服务端从 step-100 续训 ----------
say "训练服务端冷启动：--resume $RESUME ctx 12288"
nohup "$SRV/rwkv_miss_tune_server" --model "$MODEL" --resume "$RESUME" \
  --vocab "$SRV/rwkv_vocab_v20230424.txt" --ctx 12288 --chunk 128 --max-batch 256 \
  --cache-capacity 640 --keep-versions 2 --lr 2e-6 --output rl-out/server > server-sft2-r2.log 2>&1 &
for _ in $(seq 1 60); do
  sleep 5
  grep -q ready server-sft2-r2.log 2>/dev/null && break
  pgrep -f rwkv_miss_tune_server >/dev/null || die "服务端进程退出了，看 server-sft2-r2.log"
done
curl -s -m 5 "$SERVER/health" | grep -q policy_version || die "服务端 health 无响应"
STEP=$(curl -s "$SERVER/health" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['step'])")
[ "$STEP" = 100 ] || die "服务端 step=$STEP，不是 100（resume 没生效？看 server-sft2-r2.log）"
say "服务端就绪：$(curl -s "$SERVER/health" | head -c 300)"

# ---------- 4) 筛题 + GRPO ----------
FILTER_OUT=outputs/filter/sft2-s100-e5t3 RUN_OUT=outputs/rl/sft2-grpo-r2 \
  DATA=$DATA RL_LOG=outputs/rl-sft2-grpo-r2.log \
  bash chain_v5.sh
say "chain_sft2_r2 结束"
