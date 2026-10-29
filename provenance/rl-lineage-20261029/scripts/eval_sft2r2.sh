#!/usr/bin/env bash
# SFT v2 第二段 RL（sft2-grpo-r2）的中途评测：按榜单协议测七集抽样，和起点 sft2-rl-s100 对比
# 训练没停也能跑：推理服务用 --decode-batch-size 256 和训练同卡，少占显存
# 用法（服务器包根目录）：bash scripts/eval_sft2r2.sh 150 2>&1 | tee -a outputs/eval-sft2r2.log
#   参数是服务端 step（客户端 step + 100）：客户端 50 步 = 150，100 步 = 200
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
API=http://127.0.0.1:8100
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
STEP=${1:?用法：bash scripts/eval_sft2r2.sh <服务端 step，如 150>}
DECODE_BS=${DECODE_BS:-256}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 本次 resume 之后新开的 run 前缀里找 step 目录（排除 d53cd0ac… 起点和崩掉那次的 db539ef3…）
CK=$(ls -dt rl-out/server/*_1-step-"$STEP" 2>/dev/null | grep -v -e d53cd0acfc854114 -e db539ef39bccb009 | head -1)
[ -n "$CK" ] && [ -e "$CK/adapter.pth" ] || { say "FATAL: 找不到 step-$STEP 的 adapter（训练到了吗？ls -dt rl-out/server/*step-$STEP）"; exit 1; }
say "评测 $CK"

if ! curl -s -m 3 "$API/v1/adapters" >/dev/null; then
  say "起推理服务 8100（decode-batch-size $DECODE_BS）"
  screen -dmS infer bash -c "cd $PWD && rwkv_lighting_cuda/rwkv_lighting_cuda --model-path $MODEL \
    --vocab-path ./rwkv_lighting_cuda/rwkv_vocab_v20230424.txt --host 0.0.0.0 --port 8100 \
    --decode-batch-size $DECODE_BS --chunk-size 128 --state-db-path infer_sessions.db 2>&1 | tee infer.log"
  for _ in $(seq 1 60); do sleep 5; curl -s -m 3 "$API/v1/adapters" >/dev/null && break; done
  curl -s -m 3 "$API/v1/adapters" >/dev/null || { say "FATAL: 8100 没起来，看 infer.log"; exit 1; }
fi

ID=sft2r2-s$STEP
curl -sS -X POST "$API/v1/adapters" -H 'Content-Type: application/json' \
  --data "{\"adapter_id\":\"$ID\",\"path\":\"$(realpath "$CK/adapter.pth")\"}"; echo

OUT=outputs/upstream-eval/$ID-greedy-t3
say "开测 $ID -> $OUT"
$PY scripts/eval_base.py --data data/search_r1/test_upstream7_s300.parquet --format g1k --think full \
  --max_turns 4 --final_answer_turn --max_obs_tokens 500 --greedy --max_turn_tokens 2048 --topk 3 \
  --api $API --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64 \
  --adapter_id "$ID" --out "$OUT" > "$OUT.log" 2>&1 || { say "FAIL，看 $OUT.log"; exit 1; }

$PY - outputs/upstream-eval/sft2-rl-s100-greedy-t3/report.json "$OUT/report.json" <<'PYEOF'
import json, sys
base, new = (json.load(open(p))["report"] for p in sys.argv[1:])
sets = [k for k in new if k != "all"]
print(f"{'集':<16}{'s100':>8}{'新':>8}{'差':>8}")
for k in sets:
    b, n = base[k]["acc@1(EM)"], new[k]["acc@1(EM)"]
    print(f"{k:<16}{b:8.3f}{n:8.3f}{n - b:+8.3f}")
mb = sum(base[k]["acc@1(EM)"] for k in sets) / len(sets)
mn = sum(new[k]["acc@1(EM)"] for k in sets) / len(sets)
print(f"{'七集平均':<14}{mb:8.3f}{mn:8.3f}{mn - mb:+8.3f}")
a = new["all"]
print(f"format_ok {a['format_ok']}  avg_search {a['avg_search']}  end {a['end']}")
PYEOF
say "完成"
