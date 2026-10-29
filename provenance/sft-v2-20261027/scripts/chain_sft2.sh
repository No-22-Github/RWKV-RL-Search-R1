#!/usr/bin/env bash
# SFT v2 → RL 过夜链（Pro6000，放在包根目录 /mnt/nvme0n1/no22/rl-sr1-v2 下跑）
#
#   1) SFT v2：两档 lr（3e-5 / 5e-5）串行训练（train-miss-sft2.sh；并行会和 e5 + 推理服务一起爆显存）
#   2) 两档按榜单协议评七集抽样，七集平均高的进 RL；低于 GATE_MIN 就停（说明数据或训练出了问题）
#   3) 停推理服务 8100 腾显存，训练服务端 8848 从选中的 SFT adapter 冷启动（--ctx 12288）
#   4) 调 chain_v5.sh：榜单协议环境（top-3、截 500、4 次检索 + 1 轮只答）从 SFT v2 筛题 → 建池 → GRPO 200 步
#
# 前提：8002 e5 在线；8100 推理服务在线（评测用）；8848 没有服务在跑。
# 用法（服务器，screen 里）：
#   bash chain_sft2.sh 2>&1 | tee -a outputs/sft2-chain.log
# 可调：GATE_MIN（默认 0.30，v1 SFT 是 0.318）STEPS（默认 200）TARGET_MIXED（默认 600）POOL_TARGET（默认 500）
# 早上看：outputs/sft2-chain.log 的选档结果、outputs/sft2-select.json、outputs/rl/sft2-grpo/metrics.jsonl
#
# 明早评 RL 检查点要重起 8100（和之前的启动命令相同，见本文件 INFER_CMD），注册会丢，需重新注册。

set -u
[[ "${BASH_SOURCE[0]}" == */* ]] && cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd)
PY=.venv/bin/python
API=http://127.0.0.1:8100
RETRIEVER=http://127.0.0.1:8002
SERVER=http://127.0.0.1:8848
TUNE_DIR=/mnt/nvme0n1/no22/rwkv_lighting_cuda
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
SRV=$ROOT/rwkv_lighting_cuda
RUNS="sft2-r16-lr3e-5-e2 sft2-r16-lr5e-5-e2"
GATE_MIN=${GATE_MIN:-0.30}
export STEPS=${STEPS:-200}
# 筛题只跑一段（1600 题约 2.5 小时）就建池：v4 筛到 1500 mixed 用了 6.7 小时，过夜来不及。
# 池约 600–900 题，200 步每题会被抽到 4–5 次；要补池就停训、调大 TARGET_MIXED 重跑 chain_v5.sh 续筛
export TARGET_MIXED=${TARGET_MIXED:-600}
export POOL_TARGET=${POOL_TARGET:-500}
UP7=data/search_r1/test_upstream7_s300.parquet
INFER_CMD="./rwkv_lighting_cuda/rwkv_lighting_cuda --model-path $MODEL --vocab-path ./rwkv_lighting_cuda/rwkv_vocab_v20230424.txt --host 0.0.0.0 --port 8100 --decode-batch-size 1024 --chunk-size 128 --state-db-path infer_sessions.db"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }

# ---------- 0) 前置检查 ----------
curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1}' >/dev/null || die "检索 $RETRIEVER 不在"
curl -s -m 8 "$API/v1/adapters" >/dev/null || die "推理服务 $API 不在（评测要用）"
curl -s -m 3 "$SERVER/health" >/dev/null && die "8848 已有服务在跑，先停掉"
[ -x "$TUNE_DIR/train-miss-sft2.sh" ] || die "$TUNE_DIR/train-miss-sft2.sh 不在"
[ -e data/search_r1/train_rl_sft2.parquet ] || die "RL 题源 data/search_r1/train_rl_sft2.parquet 不在"
say "前置检查通过"

# ---------- 1) SFT v2（串行，前台） ----------
say "SFT 开始：$RUNS"
(cd "$TUNE_DIR" && GPUS=0 SERIAL_CHILD=1 ./train-miss-sft2.sh) || die "train-miss-sft2.sh 异常退出"
mkdir -p adapters
for r in $RUNS; do
  src=$TUNE_DIR/train-miss-sft2/$r/adapter-final.pth
  [ -e "$src" ] || die "$r 没有 adapter-final.pth，看 $TUNE_DIR/train-miss-sft2/logs/$r.log"
  cp "$src" "adapters/$r.pth"
  say "$r 完成：$(sha256sum "adapters/$r.pth" | cut -c1-16)  $(grep -o 'loss=[0-9.]*' "$TUNE_DIR/train-miss-sft2/logs/$r.log" | tail -1)"
done

# ---------- 2) 榜单协议评测、选档 ----------
COMMON="--data $UP7 --format g1k --think full --max_turns 4 --final_answer_turn --max_obs_tokens 500 --greedy
  --max_turn_tokens 2048 --topk 3 --api $API --retriever $RETRIEVER --batch_size 16 --workers 64"
mkdir -p outputs/upstream-eval
for r in $RUNS; do
  curl -sS -X POST "$API/v1/adapters" -H 'Content-Type: application/json' \
    --data "{\"adapter_id\":\"$r\",\"path\":\"$ROOT/adapters/$r.pth\"}" >/dev/null || die "注册 $r 失败"
  out=outputs/upstream-eval/$r-greedy-t3
  say "评测 $r → $out"
  $PY scripts/eval_base.py $COMMON --adapter_id "$r" --out "$out" > "$out.log" 2>&1 || die "评测 $r 失败，看 $out.log"
done
read -r BEST BEST_AVG <<< "$("$PY" - $RUNS <<'PYEOF'
import json, sys
res = {}
for r in sys.argv[1:]:
    rep = json.load(open(f"outputs/upstream-eval/{r}-greedy-t3/report.json"))["report"]
    per = {k: v["acc@1(EM)"] for k, v in rep.items() if k != "all"}
    res[r] = {"avg7": round(sum(per.values()) / len(per), 4), "per_source": per,
              "format_ok": rep["all"]["format_ok"], "avg_search": rep["all"]["avg_search"]}
    print(f"  {r}: 七集平均 {res[r]['avg7']}  format_ok {res[r]['format_ok']}  检索 {res[r]['avg_search']}  {per}", file=sys.stderr)
best = max(res, key=lambda r: res[r]["avg7"])
json.dump({"best": best, "runs": res}, open("outputs/sft2-select.json", "w"), ensure_ascii=False, indent=1)
print(best, res[best]["avg7"])
PYEOF
)"
say "选中 $BEST（七集平均 $BEST_AVG；v1 SFT 0.318，闸门 0.35，硬下限 $GATE_MIN）"
"$PY" -c "import sys; sys.exit(0 if float('$BEST_AVG') >= float('$GATE_MIN') else 1)" \
  || die "七集平均低于 $GATE_MIN，不进 RL"

# ---------- 3) 停推理服务，起训练服务端 ----------
say "停推理服务 8100（明早评测再用 INFER_CMD 重起）"
screen -S infer -X quit 2>/dev/null
pkill -f "rwkv_lighting_cuda --model-path .*--port 8100" 2>/dev/null
for _ in $(seq 1 30); do curl -s -m 2 "$API/v1/adapters" >/dev/null || break; sleep 2; done
curl -s -m 2 "$API/v1/adapters" >/dev/null && die "8100 停不掉"

say "训练服务端冷启动：adapter=adapters/$BEST.pth ctx 12288"
nohup "$SRV/rwkv_miss_tune_server" --model "$MODEL" --adapter "adapters/$BEST.pth" \
  --vocab "$SRV/rwkv_vocab_v20230424.txt" --ctx 12288 --chunk 128 --max-batch 256 \
  --cache-capacity 640 --keep-versions 2 --lr 2e-6 --output rl-out/server > server-sft2.log 2>&1 &
for _ in $(seq 1 60); do
  sleep 5
  grep -q ready server-sft2.log 2>/dev/null && break
  pgrep -f rwkv_miss_tune_server >/dev/null || die "服务端进程退出了，看 server-sft2.log"
done
curl -s -m 5 "$SERVER/health" | grep -q policy_version || die "服务端 health 无响应"
say "服务端就绪：$(curl -s "$SERVER/health" | head -c 300)"

# ---------- 4) 筛题 + GRPO（chain_v5.sh，榜单协议环境） ----------
FILTER_OUT=outputs/filter/sft2-e5t3 RUN_OUT=outputs/rl/sft2-grpo \
  DATA=data/search_r1/train_rl_sft2.parquet RL_LOG=outputs/rl-sft2-grpo.log \
  bash chain_v5.sh
say "chain_sft2 结束"
