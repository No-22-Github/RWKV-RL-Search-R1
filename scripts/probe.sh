#!/usr/bin/env bash
# RL 调参探针：从指定 adapter 冷启动训练服务端 → GRPO 跑 STEPS 步 → 停服务端 → 榜单协议评测终点
# 用新版服务端（d93775e：学习率逐请求传入、前缀缓存改全局池），不传 --lr。
#
# 用法（服务器包根目录，screen 里）：
#   NAME=lr1e-5 LR=1e-5 bash scripts/probe.sh 2>&1 | tee -a outputs/probe-lr1e-5.log
# 环境变量：
#   NAME（必填）LR（必填）STEPS（默认 30）SEED（默认 1234，与 sft2-grpo-r2 同序，可逐步配对对照）
#   INIT：起点 adapter（默认 SFT v2 RL s100）  DATA：题池（默认 sft2-s100-e5t3）
#   CLIENT_EXTRA：额外传给 rl_grpo.py 的参数（如 "--f1_weight 0 --no_std_norm"）
#   EVAL=0 跳过评测
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NAME=${NAME:?NAME 必填}
LR=${LR:?LR 必填}
STEPS=${STEPS:-30}
SEED=${SEED:-1234}
INIT=${INIT:-rl-out/server/d53cd0acfc854114943d3b2a4e027a68_1-step-100/adapter.pth}
DATA=${DATA:-outputs/filter/sft2-s100-e5t3/filtered.parquet}
CLIENT_EXTRA=${CLIENT_EXTRA:-}
EVAL=${EVAL:-1}
[ -f .env ] && { set -a; . ./.env; set +a; }   # 地址和路径不入库，见 .env.example
BIN=${TUNE_BIN:?先在 .env 里设 TUNE_BIN（rwkv_miss_tune_server 的路径）}
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
VOCAB=rwkv_lighting_cuda/rwkv_vocab_v20230424.txt
SERVER=http://127.0.0.1:8848
RETRIEVER=http://127.0.0.1:8002
OUT=outputs/rl/probe-$NAME
SRV_OUT=rl-out/probe/$NAME
SRV_LOG=server-probe-$NAME.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }

# ---------- 0) 前置检查 ----------
[ -x "$BIN" ] || die "新版服务端 $BIN 不在"
[ -e "$INIT" ] || die "起点 adapter $INIT 不在"
[ -e "$DATA" ] || die "题池 $DATA 不在"
curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1}' >/dev/null || die "检索 $RETRIEVER 不在"
curl -s -m 3 "$SERVER/health" >/dev/null && die "8848 已有服务在跑"
[ -e "$OUT/metrics.jsonl" ] && die "$OUT 已有 metrics（换 NAME，或手动挪走旧目录）"
# 评测用的 8100 推理服务和训练抢显存/内存，训练前停掉（评测阶段会重起）
if screen -ls | grep -q '\.infer\b'; then
  say "停推理服务 screen infer"
  screen -S infer -X quit
  sleep 5
fi
say "探针 $NAME：lr=$LR steps=$STEPS seed=$SEED init=$INIT extra='$CLIENT_EXTRA'"

# ---------- 1) 训练服务端冷启动 ----------
mkdir -p "$SRV_OUT"
screen -dmS "probesrv" bash -c "cd $PWD && $BIN --model $MODEL --adapter $INIT --vocab $VOCAB \
  --ctx 12288 --chunk 128 --max-batch 256 --cache-capacity 256 --cache-memory-mb 6144 \
  --keep-versions 2 --output $SRV_OUT > $SRV_LOG 2>&1"
for _ in $(seq 1 60); do
  sleep 5
  curl -s -m 3 "$SERVER/health" | grep -q policy_version && break
done
curl -s -m 5 "$SERVER/health" | grep -q policy_version || die "服务端没起来，看 $SRV_LOG"
say "服务端就绪"

# ---------- 2) GRPO ----------
"$PY" scripts/rl_grpo.py --server "$SERVER" --retriever "$RETRIEVER" \
  --topk 3 --max_obs_tokens 500 --max_turns 5 --max_turn_tokens 1024 \
  --data "$DATA" --out "$OUT" --steps "$STEPS" --lr "$LR" --f1_weight 0.3 --seed "$SEED" \
  --checkpoint_every "$STEPS" $CLIENT_EXTRA
rc=$?
say "训练结束 rc=$rc"
"$PY" scripts/probe_diff.py "$OUT" || true

# ---------- 3) 停服务端，评测终点 ----------
screen -S probesrv -X quit
for _ in $(seq 1 30); do curl -s -m 2 "$SERVER/health" >/dev/null || break; sleep 2; done
[ "$rc" = 0 ] || die "训练异常退出，不评测"
CK=$(ls -dt "$SRV_OUT"/*_1-step-"$STEPS" 2>/dev/null | head -1)
[ -n "$CK" ] && [ -e "$CK/adapter.pth" ] || die "找不到终点 adapter（$SRV_OUT/*step-$STEPS）"
if [ "$EVAL" = 1 ]; then
  bash scripts/eval_adapter.sh "probe-$NAME-s$STEPS" "$CK/adapter.pth"
fi
say "PROBE DONE $NAME"
